#!/usr/bin/env -S uv run --script

import argparse
import json
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import tqdm
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, default_data_collator

from common import LocalTimer, get_mem_stats, load_and_preprocess_data, rank0_first

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
    parser.add_argument(
        '--sharding-strategy',
        choices=('no_shard', 'shard_grad_op', 'full_shard'),
        default='full_shard',
    )
    parser.add_argument('--grad-accumulation-steps', default=1, type=int)
    return parser


def _find_transformer_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    candidate_paths = (
        ('gpt_neox', 'layers'),
        ('transformer', 'h'),
        ('model', 'layers'),
        ('decoder', 'layers'),
        ('layers',),
    )
    for path in candidate_paths:
        obj: Any = model
        for part in path:
            if not hasattr(obj, part):
                break
            obj = getattr(obj, part)
        else:
            if isinstance(obj, torch.nn.ModuleList):
                return list(obj)
            if isinstance(obj, (list, tuple)) and all(isinstance(x, torch.nn.Module) for x in obj):
                return list(obj)

    leaf_modules: list[torch.nn.Module] = []

    def visit(module: torch.nn.Module) -> None:
        children = list(module.children())
        if not children:
            if any(True for _ in module.parameters(recurse=False)):
                leaf_modules.append(module)
            return
        for child in children:
            visit(child)

    visit(model)
    return leaf_modules


def _wrap_with_fsdp2(model: torch.nn.Module, world_size: int, strategy: str) -> torch.nn.Module:
    device_mesh = init_device_mesh('cuda', (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    reshard_after_forward = strategy == 'full_shard'
    for block in _find_transformer_blocks(model):
        fully_shard(
            block,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model,
        mesh=device_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    return model


@record
def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915, PLR0912
    rank = int(os.getenv('RANK', '0'))
    local_rank = rank % torch.cuda.device_count()
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)

    logging.basicConfig(
        format=f'[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s',
        level=logging.INFO,
    )

    LOGGER.debug(os.environ)
    LOGGER.debug(args)
    LOGGER.debug(f'local_rank={local_rank} rank={rank} world_size={world_size}')

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    with rank0_first():
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model: torch.nn.Module = AutoModelForCausalLM.from_config(config, dtype=dtype)

    LOGGER.info(f'Training {sum(p.numel() for p in model.parameters())} model parameters')
    is_ddp = args.sharding_strategy == 'no_shard' or world_size == 1
    if is_ddp:
        model = model.to(device=device, dtype=dtype)
        model = torch.compile(model)  # type: ignore[assignment]
        LOGGER.info(f'Initialized model uses {get_mem_stats(device)["curr_alloc_gb"]}gb')
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                bucket_cap_mb=500,
                gradient_as_bucket_view=True,
            )
            LOGGER.info(f'After DDP: model uses {get_mem_stats(device)["curr_alloc_gb"]}gb')
        optimizer: torch.optim.Optimizer = ZeroRedundancyOptimizer(
            model.parameters(),  # type: ignore[arg-type]
            optimizer_class=torch.optim.AdamW,
            lr=args.lr,
        )
    else:
        model = _wrap_with_fsdp2(model, world_size=world_size, strategy=args.sharding_strategy)
        model = torch.compile(model)  # type: ignore[assignment]
        LOGGER.info(f'After FSDP2: model uses {get_mem_stats(device)["curr_alloc_gb"]}gb')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with rank0_first():
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
        LOGGER.debug(f'{len(train_data)} training samples. {eval_data} eval samples')

    train_sampler = (
        DistributedSampler(train_data, shuffle=True, drop_last=True) if world_size > 1 else None
    )
    eval_sampler = (
        DistributedSampler(eval_data, shuffle=False, drop_last=True) if world_size > 1 else None
    )

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
        sampler=train_sampler,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
    )
    LOGGER.info(f'{len(dataloader)} train batches per epoch, {len(eval_dataloader)} eval batches per epoch')

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,
        eta_min=args.lr * 1e-2,
    )

    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name

    state = {
        'epoch': 0,
        'global_step': 0,
        'epoch_step': 0,
        'running_loss': 0.0,
    }

    if is_experiment and (exp_dir / 'state.json').exists():
        def _load_to_device(p: str | Path) -> dict[str, Any]:
            return torch.load(p, map_location=device, weights_only=True)

        model.load_state_dict(_load_to_device(exp_dir / 'model.pt'))
        optimizer.load_state_dict(_load_to_device(exp_dir / 'optimizer.pt'))
        lr_scheduler.load_state_dict(_load_to_device(exp_dir / 'lr_scheduler.pt'))
        with (exp_dir / 'state.json').open() as f:
            state = json.load(f)
        LOGGER.info(f'Resumed! State: {state}')
    elif is_experiment:
        LOGGER.info('Creating experiment root directory')
        exp_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    timers = {k: LocalTimer(device) for k in ['data', 'forward', 'backward', 'update']}

    for state['epoch'] in range(state['epoch'], args.num_epochs):  # noqa: B020
        LOGGER.info(f'Begin epoch {state["epoch"]} at step {state["epoch_step"]}')
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(state['epoch'])

        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=rank != 0)
        start_micro_step = state['epoch_step'] * args.grad_accumulation_steps
        if start_micro_step > 0:
            progress_bar.update(start_micro_step)

        last_update_micro_step = start_micro_step
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers['data'], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < start_micro_step:
                continue

            do_update = ((i_step + 1) % args.grad_accumulation_steps == 0) or (i_step + 1 == len(dataloader))
            if not is_ddp:
                model.set_requires_gradient_sync(do_update or args.grad_accumulation_steps == 1, recurse=True)  # type: ignore[attr-defined]
            sync_ctx = model.no_sync() if (is_ddp and not do_update and args.grad_accumulation_steps > 1) else nullcontext()

            with sync_ctx:
                with timers['forward']:
                    outputs = model(**batch)

                del batch

                with timers['backward']:
                    loss = outputs.loss / args.grad_accumulation_steps
                    loss.backward()

            if do_update:
                with timers['update']:
                    optimizer.step()
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                state['global_step'] += 1
                state['epoch_step'] += 1
                state['running_loss'] += outputs.loss.item()
                progress_bar.update(i_step + 1 - last_update_micro_step)
                last_update_micro_step = i_step + 1

                if state['global_step'] % args.log_freq == 0:
                    tok_per_step = args.batch_size * args.seq_length * args.grad_accumulation_steps * world_size
                    ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                    running_loss = torch.tensor(state['running_loss'], device=device, dtype=torch.float64)
                    if world_size > 1:
                        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                    info = {
                        'global_step': state['global_step'],
                        'lr': lr_scheduler.get_last_lr()[0],
                        'running_loss': running_loss.item() / (args.log_freq * world_size),
                        'epoch': state['epoch'],
                        'epoch_progress': progress_bar.n / len(dataloader),
                        'num_batches_remaining': len(dataloader) - progress_bar.n,
                        **get_mem_stats(device),
                        'tokens_per_s': 1000 * tok_per_step / ms_per_step,
                        'time/total': ms_per_step,
                        **{f'time/{k}': timer.avg_elapsed_ms() for k, timer in timers.items()},
                    }
                    if rank == 0:
                        LOGGER.info(info)
                    torch.cuda.reset_peak_memory_stats(device)
                    state['running_loss'] = 0.0
                    for t in timers.values():
                        t.reset()

                if is_experiment and state['global_step'] % args.ckpt_freq == 0:
                    if hasattr(optimizer, 'consolidate_state_dict'):
                        optimizer.consolidate_state_dict(to=0)  # type: ignore[call-arg]
                    if rank == 0:
                        LOGGER.info('Saving checkpoint.')
                        torch.save(optimizer.state_dict(), exp_dir / 'optimizer.pt')
                        torch.save(model.state_dict(), exp_dir / 'model.pt')
                        torch.save(lr_scheduler.state_dict(), exp_dir / 'lr_scheduler.pt')
                        with (exp_dir / 'state.json').open('w') as fp:
                            json.dump(state, fp)
                    dist.barrier()

        model.eval()
        eval_loss_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        eval_token_count = torch.tensor(0.0, device=device, dtype=torch.float64)
        for _, batch in enumerate(eval_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device=device)
            with torch.no_grad():
                outputs = model(**batch)
            batch_tokens = batch['input_ids'].shape[0] * args.seq_length
            eval_loss_sum += outputs.loss.detach().to(torch.float64) * batch_tokens
            eval_token_count += batch_tokens

        if world_size > 1:
            dist.all_reduce(eval_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_token_count, op=dist.ReduceOp.SUM)

        if rank == 0:
            eval_loss = eval_loss_sum.item() / max(eval_token_count.item(), 1.0)
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float('inf')
            LOGGER.info(f'epoch {state["epoch"]}: perplexity: {perplexity} eval_loss: {eval_loss}')

        dist.barrier()
        state['epoch_step'] = 0


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
