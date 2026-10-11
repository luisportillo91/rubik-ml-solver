"""InfoNCE contrastive loss for Rubik's cube representation learning."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Information Noise-Contrastive Estimation) loss.

    Encourages anchor and positive embeddings to have high cosine similarity,
    while pushing anchor and negative embeddings apart beyond the margin.
    """

    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = False):
        super().__init__()
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

    def forward(
        self,
        anchor_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_embs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE loss.

        Args:
            anchor_emb: (B, D) anchor representation tensor.
            pos_emb: (B, D) positive representation tensor (1 move away).
            neg_embs: (B, K, D) negative representations tensor (> d moves away).

        Returns:
            Tuple of:
                - loss: scalar torch.Tensor loss.
                - accuracy: scalar torch.Tensor top-1 contrastive accuracy (0.0 to 1.0).
                - metrics: Dict with 'pos_sim', 'neg_sim', 'margin', and 'top5_acc'.
        """
        B, D = anchor_emb.shape
        # Normalize embeddings to unit hypersphere
        anchor_norm = F.normalize(anchor_emb, p=2, dim=-1)  # (B, D)
        pos_norm = F.normalize(pos_emb, p=2, dim=-1)        # (B, D)
        neg_norm = F.normalize(neg_embs, p=2, dim=-1)        # (B, K, D)

        # Positive cosine similarity: (B, 1)
        pos_sim = (anchor_norm * pos_norm).sum(dim=-1, keepdim=True)

        # Negative cosine similarity: (B, K)
        neg_sim = torch.bmm(neg_norm, anchor_norm.unsqueeze(-1)).squeeze(-1)

        # Scale by temperature: logits shape (B, 1 + K)
        logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.temperature

        # Ground truth is index 0 (positive pair)
        targets = torch.zeros(B, dtype=torch.long, device=anchor_emb.device)

        loss = F.cross_entropy(logits, targets)
        acc = (logits.argmax(dim=-1) == 0).float().mean()

        k_top = min(5, logits.size(-1))
        _, topk_indices = logits.topk(k_top, dim=-1)
        top5_acc = (topk_indices == 0).any(dim=-1).float().mean()

        with torch.no_grad():
            mean_pos_sim = pos_sim.mean().item()
            mean_neg_sim = neg_sim.mean().item()
            margin = mean_pos_sim - mean_neg_sim

        metrics = {
            "pos_sim": mean_pos_sim,
            "neg_sim": mean_neg_sim,
            "margin": margin,
            "accuracy": acc.item(),
            "top5_acc": top5_acc.item(),
        }

        return loss, acc, metrics
