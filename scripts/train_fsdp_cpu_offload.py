#!/usr/bin/env python3
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
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, default_data_collator

from common import LocalTimer, get_mem_stats, load_and_preprocess_data, rank0_first

LOGGER = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", default=None)
    parser.add_argument("-d", "--dataset-name", default=None, required=True)
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("-m", "--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=10, type=int)
    parser.add_argument("--seq-length", default=1024, type=int)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    return parser


def _wrap_pythia_gpt_neox(
    model: torch.nn.Module,
    *,
    cpu_offload: bool,
) -> torch.nn.Module:
    fsdp_kwargs: dict[str, Any] = {}
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    gpt_neox = getattr(model, "gpt_neox", None)
    for layer in gpt_neox.layers:
        fully_shard(layer, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)
    return model


@record
def main(args: argparse.Namespace) -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = rank % torch.cuda.device_count()
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )
    LOGGER.debug(os.environ)
    LOGGER.debug(args)
    LOGGER.debug(f"local_rank={local_rank} rank={rank} world_size={world_size}")

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    LOGGER.info(f"Training {sum(p.numel() for p in model.parameters())} model parameters")

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    model = _wrap_pythia_gpt_neox(model, cpu_offload=args.cpu_offload)

    LOGGER.info(f"After FSDP2: model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    with rank0_first():
        data = load_and_preprocess_data(
            args.model_name,
            args.seq_length,
            args.dataset_name,
            args.dataset_subset,
            config,
        )
        data = data.train_test_split(test_size=0.05, seed=args.seed)
        train_data = data["train"]
        eval_data = data["test"]

    LOGGER.debug(f"{len(train_data)} training samples. {len(eval_data)} eval samples")

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
    )
    LOGGER.info(f"{len(dataloader)} train batches per epoch, {len(eval_dataloader)} eval batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0.0,
    }

    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")
        model.train()

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        batches = iter(dataloader)
        for i_step in range(len(dataloader)):
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state["epoch_step"]:
                continue

            do_update = ((i_step + 1) % args.grad_accumulation_steps == 0) or (
                i_step + 1 == len(dataloader)
            )
            model.set_requires_gradient_sync(do_update)

            with timers["forward"]:
                outputs = model(**batch)
            del batch

            with timers["backward"]:
                loss = outputs.loss / args.grad_accumulation_steps
                loss.backward()

            if do_update:
                with timers["update"]:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                state["global_step"] += 1
                state["running_loss"] += outputs.loss.item()
                progress_bar.update(args.grad_accumulation_steps)

                if state["global_step"] % args.log_freq == 0:
                    tok_per_step = (
                        args.batch_size
                        * args.seq_length
                        * args.grad_accumulation_steps
                        * world_size
                    )
                    ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                    info = {
                        "global_step": state["global_step"],
                        "lr": lr_scheduler.get_last_lr()[0],
                        "running_loss": state["running_loss"] / args.log_freq,
                        "epoch": state["epoch"],
                        "epoch_progress": state["epoch_step"] / len(dataloader),
                        "num_batches_remaining": len(dataloader) - i_step,
                        **get_mem_stats(device),
                        "tokens_per_s": 1000 * tok_per_step / ms_per_step,
                        "time/total": ms_per_step,
                        **{f"time/{k}": timer.avg_elapsed_ms() for k, timer in timers.items()},
                    }
                    LOGGER.info(info)
                    torch.cuda.reset_peak_memory_stats(device)
                    state["running_loss"] = 0.0
                    for t in timers.values():
                        t.reset()

            state["epoch_step"] += 1

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
            perplexity = float("inf")
        LOGGER.info(f"epoch {state['epoch']}: perplexity: {perplexity} eval_loss: {eval_loss}")
        state["epoch_step"] = 0


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
