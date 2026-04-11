#!/usr/bin/env -S uv run --script

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as DCP  # noqa: N812
import torch.distributed.tensor.parallel as tp
import tqdm
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import DTensor, Replicate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    default_data_collator,
)

from common import LocalTimer, get_mem_stats, load_and_preprocess_data, rank_ordered
from common.llama import ModelArgs, Transformer

LOGGER = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment-name', default=None)
    parser.add_argument('-d', '--dataset-name', default=None, required=True)
    parser.add_argument('--dataset-subset', default=None)
    parser.add_argument('-m', '--model-name', default=None, required=True)
    parser.add_argument('--save-dir', default='outputs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--ckpt-freq', default=500, type=int)
    parser.add_argument('-s', '--seq-length', default=1024, type=int)
    # NEW: TP degree. Default = all GPUs on one node => pure TP, no DP replication.
    # Set --tp 1 to get behaviour equivalent to plain DDP.
    parser.add_argument('--tp', default=None, type=int, help='Tensor parallel degree. Defaults to gpus_per_node.')
    return parser


def log_tp_sharding(model, mesh_2d):  # noqa: ANN001, ANN201
    tp_mesh = mesh_2d['tp']
    tp_rank = tp_mesh.get_local_rank()
    tp_world_size = tp_mesh.size()

    attn0 = model.layers[0].attention

    def log_weight(name, w):  # noqa: ANN001, ANN202
        if isinstance(w, DTensor):
            LOGGER.info(
                '[TP] %s: rank=%d/%d type=%s global=%s local=%s',
                name,
                tp_rank,
                tp_world_size,
                type(w),
                tuple(w.shape),
                tuple(w.to_local().shape),
            )
        else:
            LOGGER.info(
                '[TP] %s: rank=%d/%d type=%s (NOT DTensor) shape=%s',
                name,
                tp_rank,
                tp_world_size,
                type(w),
                tuple(w.shape),
            )

    log_weight('attention.wq.weight', attn0.wq.weight)
    log_weight('attention.wk.weight', attn0.wk.weight)
    log_weight('attention.wv.weight', attn0.wv.weight)
    log_weight('attention.wo.weight', attn0.wo.weight)


@record
def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915, PLR0912
    gpus_on_node = torch.cuda.device_count()
    rank = int(os.getenv('RANK', '0'))
    local_rank = rank % gpus_on_node
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)
    assert world_size % gpus_on_node == 0, 'Все ноды должны иметь одинаковое число GPU'

    # ------------------------------------------------------------------
    # NEW: 2D Device Mesh  (dp_size x tp_size)
    # ------------------------------------------------------------------
    # tp_size  — сколько GPU шэрят один слой (внутри ноды, по NVLink)
    # dp_size  — сколько реплик (между нодами, по InfiniBand / Ethernet)
    #
    # Пример: 2 ноды x 8 GPU, --tp 8  =>  dp=2, tp=8
    #         1 нода x 8 GPU, --tp 4  =>  dp=2, tp=4
    #         2 ноды x 8 GPU, --tp 4  =>  dp=4, tp=4
    tp_size = args.tp if args.tp is not None else gpus_on_node
    dp_size = world_size // tp_size
    assert world_size % tp_size == 0, f'world_size={world_size} должен делиться на --tp={tp_size}'
    mesh = dist.device_mesh.init_device_mesh(
        'cuda',
        (dp_size, tp_size),
        mesh_dim_names=('dp', 'tp'),
    )

    logging.basicConfig(
        format=f'[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s',
        level=logging.INFO,
    )

    LOGGER.info(
        f'local_rank={local_rank} rank={rank} world_size={world_size} dp_size={mesh["dp"].size()} tp_size={mesh["tp"].size()}'  # noqa: E501
    )

    torch.manual_seed(args.seed)

    with rank_ordered(should_go_first=local_rank == 0), device:
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model_args = ModelArgs(
            dim=config.hidden_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            vocab_size=config.vocab_size,
            max_seq_len=args.seq_length,
        )
        model = Transformer(model_args)
    LOGGER.info(f'Training {sum(p.numel() for p in model.parameters())} model parameters')

    # ------------------------------------------------------------------
    # NEW: Tensor Parallelism — разрезаем веса по оси TP
    # ------------------------------------------------------------------
    # ColwiseParallel: шардируем по dim=0 выходного измерения.
    #   Каждый GPU держит часть «колонок» матрицы весов.
    #
    # RowwiseParallel: шардируем по dim=1 выходного измерения (строки).
    #
    # Пара ColwiseParallel → RowwiseParallel образует один «TP-юнит» без
    # лишних AllReduce между ними: всё идёт как один matmul на каждой GPU.

    if tp_size > 1:
        LOGGER.info('Applying Tensor Parallelism')

        LOGGER.info('Before TP')
        log_tp_sharding(model, mesh)

        attn0 = model.layers[0].attention
        assert attn0.n_heads % tp_size == 0, 'n_heads must be divisible by tp_size'
        assert attn0.n_kv_heads % tp_size == 0, 'n_kv_heads must be divisible by tp_size'

        # План шардирования для блока TransformerBlock:
        layer_tp_plan = {
            # Attention: q/k/v — Colwise: шардирование по выходным каналам (головы)
            'attention.wq': tp.ColwiseParallel(use_local_output=True),
            'attention.wk': tp.ColwiseParallel(use_local_output=True),
            'attention.wv': tp.ColwiseParallel(use_local_output=True),
            # Выход attention: Rowwise: собирает локальные головы, делает AllReduce
            'attention.wo': tp.RowwiseParallel(),
            # MLP: w1/w3 — Colwise, w2 — Rowwise
            'feed_forward.w1': tp.ColwiseParallel(use_local_output=True),
            'feed_forward.w2': tp.RowwiseParallel(),
            'feed_forward.w3': tp.ColwiseParallel(use_local_output=True),
        }

        # 1) Настраиваем каждый блок: локальное количество голов
        for block in model.layers:
            attn = block.attention
            attn.n_heads = attn.n_heads // tp_size
            attn.n_kv_heads = attn.n_kv_heads // tp_size
            attn.n_rep = attn.n_heads // attn.n_kv_heads
            tp.parallelize_module(
                module=block,
                device_mesh=mesh['tp'],
                parallelize_plan=layer_tp_plan,
            )

        # 2) Embedding и выходной слой
        model = tp.parallelize_module(
            model,
            mesh['tp'],
            {
                # эмбеддинги: Rowwise — шардирование по embedding‑вектору,
                # вход реплицирован, выход реплицирован (без SP для простоты)
                'tok_embeddings': tp.RowwiseParallel(
                    input_layouts=Replicate(),
                ),
                # выходной линейный: Colwise — делим логицы по vocab‑оси,
                # оставляем выход реплицированным (не используем loss_parallel здесь)
                'output': tp.ColwiseParallel(
                    output_layouts=Replicate(),
                ),
            },
        )

        LOGGER.info('After TP')
        log_tp_sharding(model, mesh)

    model = model.to_empty(device=device)
    model.init_weights()
    model.train()
    LOGGER.info(f'Initialized model uses: {get_mem_stats(device)["curr_alloc_gb"]:.3f} gb')

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    with rank_ordered(should_go_first=local_rank == 0):
        data = load_and_preprocess_data(
            args.model_name,
            args.seq_length,
            args.dataset_name,
            args.dataset_subset,
            config,
        )
    data = data.train_test_split(test_size=0.05, seed=args.seed)
    train_data = data['train']
    eval_data = data['test']

    # NEW: DistributedSampler теперь работает только по DP-измерению.
    # Все GPU в одной TP-группе должны получать ОДИНАКОВЫЙ батч,
    # поэтому num_replicas=dp_size, rank=dp_rank
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
        sampler=DistributedSampler(
            train_data,
            shuffle=True,
            drop_last=True,
            num_replicas=mesh['dp'].size(),  # число DP-реплик
            rank=mesh['dp'].get_local_rank(),  # индекс этой реплики
        ),
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
    )
    LOGGER.info(f'{len(dataloader)} train batches per epoch, {len(eval_dataloader)} eval batches per epoch')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=args.lr * 1e-2)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name

    state = {
        'epoch': 0,
        'global_step': 0,
        'epoch_step': 0,
        'running_loss': 0,
    }
    if is_experiment and (exp_dir / 'state.json').exists():

        def _load_to_device(p: str | Path) -> dict[str, Any]:
            return torch.load(p, map_location=device, weights_only=True)

        DCP.load(
            dict(model=model, optimizer=optimizer),
            checkpoint_id=exp_dir / 'checkpoint',
        )

        lr_scheduler.load_state_dict(_load_to_device(exp_dir / 'lr_scheduler.pt'))
        with (exp_dir / 'state.json').open() as f:
            state = json.load(f)
        LOGGER.info(f'Resumed! State: {state}')

    dist.barrier()
    if is_experiment and rank == 0:
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    timers = {k: LocalTimer(device) for k in ['data', 'forward', 'backward', 'update']}

    for state['epoch'] in range(state['epoch'], args.num_epochs):  # noqa: B020
        LOGGER.info(f'Begin epoch {state["epoch"]} at step {state["epoch_step"]}')
        model.train()

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state['epoch_step'] > 0:
            progress_bar.update(state['epoch_step'])

        dataloader.sampler.set_epoch(state['epoch'])  # type: ignore[attr-defined]
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers['data'], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state['epoch_step']:
                continue

            with timers['forward']:
                input_ids = batch['input_ids']
                labels = batch['labels']
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(0, 1),
                )
                del batch

            with timers['backward']:
                loss.backward()

            with timers['update']:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            state['global_step'] += 1
            state['epoch_step'] += 1
            state['running_loss'] += loss.item()
            progress_bar.update(1)

            if state['global_step'] % args.log_freq == 0:
                # scale to mesh["dp"] to get effective batch_size
                tok_per_step = mesh['dp'].size() * args.batch_size * args.seq_length
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                info = {
                    'global_step': state['global_step'],
                    'lr': lr_scheduler.get_last_lr()[0],
                    'running_loss': state['running_loss'] / args.log_freq,
                    'epoch': state['epoch'],
                    'epoch_progress': state['epoch_step'] / len(dataloader),
                    'num_batches_remaining': len(dataloader) - i_step,
                    **get_mem_stats(device),
                    'tokens_per_s': 1000 * tok_per_step / ms_per_step,
                    'dp_size': dp_size,
                    'tp_size': tp_size,
                    'time/total': ms_per_step,
                    **{f'time/{k}': timer.avg_elapsed_ms() for k, timer in timers.items()},
                }
                LOGGER.info(info)

                torch.cuda.reset_peak_memory_stats(device)
                state['running_loss'] = 0
                for t in timers.values():
                    t.reset()

            if is_experiment and state['global_step'] % args.ckpt_freq == 0:
                LOGGER.info('Saving checkpoint.')
                dist.barrier()
                # NOTE: we have to call this on ALL ranks
                DCP.save(
                    dict(model=model, optimizer=optimizer),
                    checkpoint_id=exp_dir / 'checkpoint',
                )
                if rank == 0:
                    torch.save(lr_scheduler.state_dict(), exp_dir / 'lr_scheduler.pt')
                    with (exp_dir / 'state.json').open('w') as fp:
                        json.dump(state, fp)
                dist.barrier()

        # ------------------------------------------------------------------
        # Eval
        # Теперь не можем запускать eval только на rank=0, потому что модель теперь шардирована:
        # у rank=0 нет полной модели, только ее часть
        # ------------------------------------------------------------------
        dist.barrier()

        model.eval()
        losses = []
        for _, batch in enumerate(eval_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device=device)
            with torch.no_grad():
                input_ids = batch['input_ids']
                labels = batch['labels']
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(0, 1),
                )
            losses.append(loss.item())

        eval_loss = torch.mean(torch.tensor(losses))
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')
        LOGGER.info(f'epoch {state["epoch"]}: perplexity={perplexity:.2f} eval_loss={eval_loss:.4f}')

        dist.barrier()

        state['epoch_step'] = 0


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
