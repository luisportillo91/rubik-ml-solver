#!/usr/bin/env python3
"""Large-scale step-based training script for Rubik's Cube Transformer Encoder using InfoNCE loss.

Supports:
  - Multi-GPU DistributedDataParallel (DDP) via torchrun (e.g. 2x NVIDIA T4 on Kaggle)
  - Automatic Mixed Precision (AMP) with GradScaler for NVIDIA Tensor Cores
  - Asynchronous background CPU batch generation via PyTorch DataLoader workers
  - Single concatenated forward pass for maximum GPU compute utilization
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from rubik.dataset import ContrastiveCubeIterableDataset, generate_contrastive_batch
from rubik.encoder import CubeTransformerEncoder
from rubik.loss import InfoNCELoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Rubik's Cube Transformer Encoder with InfoNCE loss.")
    parser.add_argument("--steps", type=int, default=500, help="Total number of optimization steps (N)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU per step (number of anchors)")
    parser.add_argument("--log-every", type=int, default=25, help="Log progress every N steps")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--k-negatives", type=int, default=100, help="Negatives per anchor (default 100)")
    parser.add_argument("--distance-threshold", type=int, default=5, help="Excluded buffer distance threshold k (default 5)")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d-embed", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature tau")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('auto', 'cuda', or 'cpu')")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader background worker processes per GPU (default 1)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (AMP) on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--history-file", type=str, default="training_history.json", help="Path to save metric history JSON")
    return parser.parse_args()


def init_distributed():
    """Detect and initialize distributed environment if launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    is_distributed = local_rank != -1 and world_size > 1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = None  # Will be resolved based on args.device

    return is_distributed, rank, local_rank, world_size, device


def main():
    args = parse_args()
    is_distributed, rank, local_rank, world_size, dist_device = init_distributed()
    is_main_process = rank == 0

    # Determine device
    if is_distributed:
        device = dist_device
    else:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    use_cuda = device.type == "cuda"
    use_amp = args.amp and use_cuda

    # Prevent thread oversubscription on machines with limited vCPUs (e.g. 4 vCPUs)
    if is_distributed or use_cuda:
        torch.set_num_threads(1)

    effective_batch_size = args.batch_size * world_size
    total_negatives_per_step = effective_batch_size * args.k_negatives

    if is_main_process:
        print("=" * 96)
        print(" RUBIK'S CUBE INFONCE MULTI-GPU TRAINING PIPELINE")
        print(f" Environment   : {'DDP (DistributedDataParallel)' if is_distributed else 'Single-Process'}")
        print(f" World Size    : {world_size} GPU(s) | Rank: {rank} | Local Device: {device}")
        print(f" Batch Size    : {args.batch_size} / GPU -> Effective Global Batch: {effective_batch_size}")
        print(f" Negatives (K) : {args.k_negatives} / anchor -> {total_negatives_per_step:,} total negatives / step")
        print(f" AMP Precision : {'Enabled (FP16 GradScaler)' if use_amp else 'Disabled (FP32)'}")
        print(f" CPU Workers   : {args.num_workers} worker(s) per GPU process")
        print(f" Topology      : Positive = 1 move | Excluded = [2, {args.distance_threshold}] | Negatives > {args.distance_threshold}")
        print(f" Total Steps   : {args.steps} (Pure step-based, zero disk files, infinite stream)")
        print("=" * 96)

    # Initialize model
    raw_model = CubeTransformerEncoder(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 2,
        d_embed=args.d_embed,
        pooling="cls",
        dropout=0.1,
    ).to(device)

    if is_distributed:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = raw_model

    num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Model Architecture: {args.num_layers} layers, {args.nhead} heads, {args.d_model} d_model ({num_params:,} parameters)")

    criterion = InfoNCELoss(temperature=args.temperature).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")
    history: List[Dict] = []

    # Setup Streaming DataLoader
    def worker_init_fn(worker_id: int):
        # Unique seed per worker and rank to guarantee distinct scrambles
        seed = (torch.initial_seed() % (2**31)) + worker_id + rank * 10007
        random.seed(seed)

    dataset = ContrastiveCubeIterableDataset(
        batch_size=args.batch_size,
        k_negatives=args.k_negatives,
        d_threshold=args.distance_threshold,
    )

    if args.num_workers > 0:
        data_loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            worker_init_fn=worker_init_fn,
            prefetch_factor=4,
        )
        data_iter = iter(data_loader)
    else:
        data_iter = None

    if is_main_process:
        header = (
            f"{'Step':>7}/{args.steps:<5} | {'Loss':>8} | {'Top-1 Acc':>10} | {'Top-5 Acc':>10} | "
            f"{'Pos Sim':>9} | {'Neg Sim':>9} | {'Margin':>8} | {'Speed':>9}"
        )
        print("\n" + header)
        print("-" * 96)

    # Window metric accumulators
    window_loss = 0.0
    window_top1 = 0.0
    window_top5 = 0.0
    window_pos_sim = 0.0
    window_neg_sim = 0.0
    window_margin = 0.0
    window_start_time = time.perf_counter()

    model.train()
    for step in range(1, args.steps + 1):
        # 1. Fetch next contrastive batch
        if data_iter is not None:
            anchor_t, pos_t, neg_t = next(data_iter)
            anchor_t = anchor_t.to(device, non_blocking=True)
            pos_t = pos_t.to(device, non_blocking=True)
            neg_t = neg_t.to(device, non_blocking=True)
        else:
            anchor_t, pos_t, neg_t = generate_contrastive_batch(
                batch_size=args.batch_size,
                k_negatives=args.k_negatives,
                d_threshold=args.distance_threshold,
                device=device,
            )

        B, K, _ = neg_t.shape
        neg_flat = neg_t.view(B * K, 54)

        # 2. Optimized single forward pass: concatenate all cubes into one batch
        all_cubes = torch.cat([anchor_t, pos_t, neg_flat], dim=0)

        optimizer.zero_grad()

        autocast_kwargs = {"device_type": "cuda", "enabled": use_amp}
        with torch.amp.autocast(**autocast_kwargs) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=use_amp):
            all_embs, _ = model(all_cubes)
            anchor_emb = all_embs[:B]
            pos_emb = all_embs[B : 2 * B]
            neg_embs = all_embs[2 * B :].view(B, K, -1)

            # 3. Compute InfoNCE loss & metrics
            loss, acc, metrics = criterion(anchor_emb, pos_emb, neg_embs)

        # 4. Backward & optimize with GradScaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Gather metrics across ranks if distributed
        if is_distributed:
            metric_t = torch.tensor(
                [loss.item(), acc.item(), metrics.get("top5_acc", 0.0), metrics["pos_sim"], metrics["neg_sim"], metrics["margin"]],
                device=device,
            )
            dist.all_reduce(metric_t, op=dist.ReduceOp.SUM)
            metric_vals = (metric_t / world_size).cpu().tolist()
            step_loss, step_top1, step_top5, step_pos, step_neg, step_margin = metric_vals
        else:
            step_loss = loss.item()
            step_top1 = acc.item()
            step_top5 = metrics.get("top5_acc", 0.0)
            step_pos = metrics["pos_sim"]
            step_neg = metrics["neg_sim"]
            step_margin = metrics["margin"]

        # Accumulate metrics
        window_loss += step_loss
        window_top1 += step_top1
        window_top5 += step_top5
        window_pos_sim += step_pos
        window_neg_sim += step_neg
        window_margin += step_margin

        # 5. Periodic logging (main process only)
        if step % args.log_every == 0 or step == args.steps:
            steps_in_window = (step % args.log_every) if (step % args.log_every) != 0 else args.log_every
            window_elapsed = time.perf_counter() - window_start_time
            steps_per_sec = steps_in_window / window_elapsed if window_elapsed > 0 else 0.0

            avg_loss = window_loss / steps_in_window
            avg_top1 = (window_top1 / steps_in_window) * 100
            avg_top5 = (window_top5 / steps_in_window) * 100
            avg_pos = window_pos_sim / steps_in_window
            avg_neg = window_neg_sim / steps_in_window
            avg_margin = window_margin / steps_in_window

            if is_main_process:
                step_str = f"{step:>7d}/{args.steps:<5d}"
                top1_str = f"{avg_top1:6.1f}%"
                top5_str = f"{avg_top5:6.1f}%"
                margin_str = f"{avg_margin:+7.4f}"
                speed_str = f"{steps_per_sec:5.2f} stp/s"

                print(
                    f"{step_str} | {avg_loss:8.4f} | {top1_str:>10} | {top5_str:>10} | "
                    f"{avg_pos:+9.4f} | {avg_neg:+9.4f} | {margin_str:>8} | {speed_str:>9}"
                )

                history_entry = {
                    "step": step,
                    "loss": round(avg_loss, 4),
                    "top1_acc": round(avg_top1, 2),
                    "top5_acc": round(avg_top5, 2),
                    "pos_sim": round(avg_pos, 4),
                    "neg_sim": round(avg_neg, 4),
                    "margin": round(avg_margin, 4),
                    "steps_per_sec": round(steps_per_sec, 2),
                }
                history.append(history_entry)

                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(args.save_dir, "best_encoder.pt")
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_loss,
                            "args": vars(args),
                        },
                        best_path,
                    )

                # Save history JSON
                history_path = os.path.join(args.save_dir, args.history_file)
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

            # Reset window accumulators
            window_loss = 0.0
            window_top1 = 0.0
            window_top5 = 0.0
            window_pos_sim = 0.0
            window_neg_sim = 0.0
            window_margin = 0.0
            window_start_time = time.perf_counter()

        # Regular periodic checkpoints
        if step % args.save_every == 0 and is_main_process:
            step_path = os.path.join(args.save_dir, f"encoder_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                step_path,
            )

    if is_main_process:
        print("-" * 96)
        print(f"Training completed successfully. Best model saved to: {os.path.join(args.save_dir, 'best_encoder.pt')}")
        print(f"Metrics history saved to: {os.path.join(args.save_dir, args.history_file)}")
        print("=" * 96)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

