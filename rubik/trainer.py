"""High-level interactive trainer for Rubik's cube representation learning."""

from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from rubik.dataset import generate_contrastive_batch
from rubik.encoder import CubeTransformerEncoder
from rubik.loss import InfoNCELoss


@dataclass
class TrainingConfig:
    """Configuration for InfoNCE contrastive training."""
    steps: int = 1000
    batch_size: int = 32          # M anchors per step
    k_negatives: int = 64         # N canonical negatives per anchor (M x N negatives)
    distance_threshold: int = 5   # Excluded buffer zone [2, distance_threshold]
    min_scramble: int = 1
    max_scramble: int = 25
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: Optional[int] = None
    d_embed: int = 128
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    device: str = "auto"          # 'auto', 'cuda', 'cuda:0', 'cpu'
    multi_gpu: bool = True        # Use nn.DataParallel if multiple GPUs available
    amp: bool = True              # Automatic mixed precision on CUDA
    log_every: int = 25
    save_every: int = 250
    save_dir: Optional[str] = "checkpoints"
    history_file: str = "training_history.json"
    show_progress_bar: bool = True


class TrainResult:
    """Result object returned by train_infonce with history and plotting helpers."""

    def __init__(
        self,
        model: CubeTransformerEncoder,
        history: List[Dict[str, Any]],
        best_loss: float,
        best_step: int,
        config: TrainingConfig,
        best_checkpoint_path: Optional[str] = None,
    ):
        self.model = model
        self.history = history
        self.best_loss = best_loss
        self.best_step = best_step
        self.config = config
        self.best_checkpoint_path = best_checkpoint_path

    def __repr__(self) -> str:
        return (
            f"TrainResult(steps={len(self.history)}, best_loss={self.best_loss:.4f} "
            f"at step {self.best_step}, checkpoint='{self.best_checkpoint_path}')"
        )

    def plot_curves(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 9)):
        """Plot the training curves (Loss, Accuracy, Similarities, Margin)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required to plot training curves. Install with: pip install matplotlib")
            return None

        if not self.history:
            print("No training history available to plot.")
            return None

        steps = [h["step"] for h in self.history]
        losses = [h["loss"] for h in self.history]
        top1 = [h["top1_acc"] for h in self.history]
        top5 = [h["top5_acc"] for h in self.history]
        pos_sim = [h["pos_sim"] for h in self.history]
        neg_sim = [h["neg_sim"] for h in self.history]
        margins = [h["margin"] for h in self.history]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Rubik's Cube Transformer InfoNCE Training (M={self.config.batch_size}, N={self.config.k_negatives})",
            fontsize=15,
            fontweight="bold",
        )

        # 1. InfoNCE Loss
        axes[0, 0].plot(steps, losses, color="#1f77b4", lw=2)
        axes[0, 0].set_title("InfoNCE Contrastive Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Optimization Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Retrieval Accuracy
        axes[0, 1].plot(steps, top1, label="Top-1 Acc (%)", color="#2ca02c", lw=2)
        axes[0, 1].plot(steps, top5, label="Top-5 Acc (%)", color="#ff7f0e", lw=2)
        axes[0, 1].set_title("1-Step Move Retrieval Accuracy", fontweight="bold")
        axes[0, 1].set_xlabel("Optimization Step")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Cosine Similarities
        axes[1, 0].plot(steps, pos_sim, label="Positives (dist=1)", color="#2ca02c", lw=2)
        axes[1, 0].plot(steps, neg_sim, label="Negatives (dist>5)", color="#d62728", lw=2)
        axes[1, 0].set_title("Positive vs. Negative Cosine Similarity", fontweight="bold")
        axes[1, 0].set_xlabel("Optimization Step")
        axes[1, 0].set_ylabel("Cosine Similarity")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Margin
        axes[1, 1].plot(steps, margins, color="#9467bd", lw=2)
        axes[1, 1].set_title("Representation Margin (Pos - Neg)", fontweight="bold")
        axes[1, 1].set_xlabel("Optimization Step")
        axes[1, 1].set_ylabel("Margin")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Training curves saved to: {save_path}")
        plt.show()
        return fig


def train_infonce(
    config: Optional[TrainingConfig] = None,
    **kwargs,
) -> TrainResult:
    """
    Train a Rubik's Cube Transformer Encoder directly inside Python / Jupyter Notebooks.

    Features:
      - Interactive progress bar with live metrics via tqdm
      - Vectorized M x N batch generation using group translation invariance
      - Automatic GPU & multi-GPU (DataParallel) detection
      - AMP mixed precision support for NVIDIA Tensor Cores

    Args:
        config: TrainingConfig instance, or pass individual kwargs:
            steps: int (default 1000)
            batch_size: int (default 32)
            k_negatives: int (default 64)
            lr: float (default 3e-4)
            device: str ('auto', 'cuda', 'cpu')
            save_dir: Optional[str] ('checkpoints')
            ...

    Returns:
        TrainResult with the trained model, metric history, and plotting methods.
    """
    if config is None:
        cfg = TrainingConfig(**kwargs)
    else:
        # Allow overriding config with kwargs
        cfg_dict = asdict(config)
        cfg_dict.update(kwargs)
        cfg = TrainingConfig(**cfg_dict)

    # Determine Device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    use_cuda = device.type == "cuda"
    use_amp = cfg.amp and use_cuda
    num_gpus = torch.cuda.device_count() if use_cuda else 0
    use_dp = use_cuda and cfg.multi_gpu and num_gpus > 1

    # Single-thread CPU ops to prevent CPU contention
    if use_cuda:
        torch.set_num_threads(1)

    print("=" * 80)
    print(" RUBIK'S CUBE INFONCE INTERACTIVE TRAINER")
    print(f" Device        : {device} ({num_gpus} GPU(s) available, DataParallel={use_dp})")
    print(f" Configuration : M={cfg.batch_size} anchors x N={cfg.k_negatives} canonical negatives")
    print(f" Total Per Step: {cfg.batch_size * cfg.k_negatives:,} negatives strictly with dist > {cfg.distance_threshold}")
    print(f" Optimization  : {cfg.steps} steps | LR={cfg.lr} | AMP={'Enabled' if use_amp else 'Disabled'}")
    print("=" * 80)

    # Initialize model
    dim_ff = cfg.dim_feedforward if cfg.dim_feedforward is not None else cfg.d_model * 4
    raw_model = CubeTransformerEncoder(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=dim_ff,
        d_embed=cfg.d_embed,
        pooling="cls",
        dropout=cfg.dropout,
    ).to(device)

    if use_dp:
        model: nn.Module = nn.DataParallel(raw_model)
    else:
        model = raw_model

    criterion = InfoNCELoss(temperature=cfg.temperature).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.steps, eta_min=1e-6)

    # AMP Scaler
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if cfg.save_dir:
        os.makedirs(cfg.save_dir, exist_ok=True)

    best_loss = float("inf")
    best_step = 0
    best_checkpoint_path = None
    history: List[Dict[str, Any]] = []

    # Window accumulators for smoothed logging
    window_loss = 0.0
    window_top1 = 0.0
    window_top5 = 0.0
    window_pos = 0.0
    window_neg = 0.0
    window_margin = 0.0
    window_count = 0

    # Progress Bar
    pbar = tqdm(
        range(1, cfg.steps + 1),
        desc="Training InfoNCE",
        unit="step",
        disable=not cfg.show_progress_bar,
    )

    model.train()
    for step in pbar:
        # 1. Generate M x N vectorized batch
        anchor_t, pos_t, neg_t = generate_contrastive_batch(
            batch_size=cfg.batch_size,
            k_negatives=cfg.k_negatives,
            d_threshold=cfg.distance_threshold,
            min_scramble=cfg.min_scramble,
            max_scramble=cfg.max_scramble,
            device=device,
        )

        M, K, _ = neg_t.shape
        neg_flat = neg_t.view(M * K, 54)

        # 2. Single forward pass
        all_cubes = torch.cat([anchor_t, pos_t, neg_flat], dim=0)

        optimizer.zero_grad()

        autocast_kwargs = {"device_type": "cuda", "enabled": use_amp}
        with torch.amp.autocast(**autocast_kwargs) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=use_amp):
            all_embs, _ = model(all_cubes)
            anchor_emb = all_embs[:M]
            pos_emb = all_embs[M : 2 * M]
            neg_embs = all_embs[2 * M :].view(M, K, -1)

            loss, acc, metrics = criterion(anchor_emb, pos_emb, neg_embs)

        # 3. Optimize
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accumulate metrics
        loss_val = loss.item()
        top1_val = acc.item() * 100
        top5_val = metrics.get("top5_acc", 0.0) * 100
        pos_val = metrics["pos_sim"]
        neg_val = metrics["neg_sim"]
        margin_val = metrics["margin"]

        window_loss += loss_val
        window_top1 += top1_val
        window_top5 += top5_val
        window_pos += pos_val
        window_neg += neg_val
        window_margin += margin_val
        window_count += 1

        # Live Progress Bar update
        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "top1": f"{top1_val:.1f}%",
            "top5": f"{top5_val:.1f}%",
            "pos": f"{pos_val:+.3f}",
            "neg": f"{neg_val:+.3f}",
            "margin": f"{margin_val:+.3f}",
        })

        # Periodic logging / checkpointing
        if step % cfg.log_every == 0 or step == cfg.steps:
            avg_loss = window_loss / window_count
            avg_top1 = window_top1 / window_count
            avg_top5 = window_top5 / window_count
            avg_pos = window_pos / window_count
            avg_neg = window_neg / window_count
            avg_margin = window_margin / window_count

            entry = {
                "step": step,
                "loss": round(avg_loss, 4),
                "top1_acc": round(avg_top1, 2),
                "top5_acc": round(avg_top5, 2),
                "pos_sim": round(avg_pos, 4),
                "neg_sim": round(avg_neg, 4),
                "margin": round(avg_margin, 4),
            }
            history.append(entry)

            # Checkpoint best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_step = step
                if cfg.save_dir:
                    best_checkpoint_path = os.path.join(cfg.save_dir, "best_encoder.pt")
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_loss,
                            "config": asdict(cfg),
                        },
                        best_checkpoint_path,
                    )

            # Reset window
            window_loss = 0.0
            window_top1 = 0.0
            window_top5 = 0.0
            window_pos = 0.0
            window_neg = 0.0
            window_margin = 0.0
            window_count = 0

        # Periodic checkpoints
        if cfg.save_dir and step % cfg.save_every == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                },
                os.path.join(cfg.save_dir, f"encoder_step_{step}.pt"),
            )

    # Save final history JSON
    if cfg.save_dir and cfg.history_file:
        hist_path = os.path.join(cfg.save_dir, cfg.history_file)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    raw_model.eval()
    print("\n" + "=" * 80)
    print(f" Training complete! Best Loss: {best_loss:.4f} at Step {best_step}")
    if best_checkpoint_path:
        print(f" Best Checkpoint Saved : {best_checkpoint_path}")
    print("=" * 80)

    return TrainResult(
        model=raw_model,
        history=history,
        best_loss=best_loss,
        best_step=best_step,
        config=cfg,
        best_checkpoint_path=best_checkpoint_path,
    )
