#!/usr/bin/env -S uv run --script

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    default_data_collator,
)

from common import LocalTimer, get_mem_stats, load_and_preprocess_data

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
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--grad-accumulation-steps', type=int, default=1)
    return parser


def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915, PLR0912
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s:%(message)s',
        level=logging.INFO,
    )

    LOGGER.debug(os.environ)
    LOGGER.debug(args)

    device = torch.device('cuda')
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32

    torch.manual_seed(args.seed)

    # Initializing an **untrained** model
    model: torch.nn.Module
    with device:
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    LOGGER.info(f'Training {sum(p.numel() for p in model.parameters())} model parameters')
    
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    model = torch.compile(model)  # type: ignore[assignment]

    LOGGER.info(f'Initialized model uses {get_mem_stats(device)["curr_alloc_gb"]}gb')

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

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)

    # NOTE: T_max and eta_min were arbitrarily chosen
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=args.lr * 1e-2)

    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name

    # attempt resume
    state = {
        'epoch': 0,
        'global_step': 0,
        'epoch_step': 0,
        'running_loss': 0,
    }
    if is_experiment and (exp_dir / 'state.json').exists():
        # NOTE: weights_only is to protect against arbitrary code execution with pickle decoding.
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

    # will use this to understand breakdown of speed
    timers = {k: LocalTimer(device) for k in ['data', 'forward', 'backward', 'update']}

    for state['epoch'] in range(state['epoch'], args.num_epochs):  # noqa: B020
        LOGGER.info(f'Begin epoch {state["epoch"]} at step {state["epoch_step"]}')
        model.train()

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state['epoch_step'] > 0:
            progress_bar.update(state['epoch_step'])

        # NOTE: This is not standard. Normally you can just iterate directly over dataloader.
        #       We are doing this so we can explicitly measure the time it takes to generate a batch.
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            # measure the time it takes to generate a batch and move it to the GPU
            with timers['data'], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            # For resuming, this has to come after getting the next batch, so we move through the dataset properly
            if i_step < state['epoch_step']:
                continue

            with timers['forward']:
                outputs = model(**batch)
                del batch

            with timers['backward']:
                loss = outputs.loss / args.grad_accumulation_steps
                loss.backward()

            do_update = ((i_step + 1) % args.grad_accumulation_steps == 0) or (i_step + 1 == len(dataloader))

            if (do_update):
                with timers['update']:
                    optimizer.step()
                    lr_scheduler.step()
                    # NOTE: set_to_none=True will de-allocate the gradients, saving us some memory
                    optimizer.zero_grad(set_to_none=True)

                state['global_step'] += 1
                state['running_loss'] += outputs.loss.item()
                progress_bar.update(args.grad_accumulation_steps)

                if state['global_step'] % args.log_freq == 0:
                    tok_per_step = args.batch_size * args.seq_length * args.grad_accumulation_steps
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
                    torch.save(optimizer.state_dict(), exp_dir / 'optimizer.pt')
                    torch.save(model.state_dict(), exp_dir / 'model.pt')
                    torch.save(lr_scheduler.state_dict(), exp_dir / 'lr_scheduler.pt')
                    with (exp_dir / 'state.json').open('w') as fp:
                        json.dump(state, fp)
            state['epoch_step'] += 1
        model.eval()
        losses = []
        for _, batch in enumerate(eval_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device=device)

            with torch.no_grad():
                outputs = model(**batch)

            losses.append(outputs.loss.item())
        losses = torch.Tensor(losses)  # type: ignore[assignment]

        try:
            eval_loss = torch.mean(losses)  # type: ignore[call-overload]
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')

        LOGGER.info(f'epoch {state["epoch"]}: perplexity: {perplexity} eval_loss: {eval_loss}')

        state['epoch_step'] = 0


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
