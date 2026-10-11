"""Unit tests for the InfoNCE loss, streaming generator, and training loop."""

import pytest
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from rubik.cube import apply_moves, create_solved_cube
from rubik.dataset import (
    ContrastiveCubeIterableDataset,
    generate_contrastive_batch,
    generate_contrastive_sample,
)
from rubik.encoder import CubeTransformerEncoder
from rubik.loss import InfoNCELoss
from rubik.search import find_distance, is_within_distance


def test_generate_contrastive_sample_topology():
    """Verify that positive is 1 move away, and all negatives are > d moves away (k=5)."""
    d_threshold = 5
    k_negatives = 3
    anchor, pos, negs = generate_contrastive_sample(
        d_threshold=d_threshold,
        k_negatives=k_negatives,
        min_scramble=5,
        max_scramble=15,
    )

    # 1. Positive must be within distance <= 1 (specifically 1 move away)
    assert is_within_distance(anchor, pos, max_distance=1)
    pos_dist = find_distance(anchor, pos, max_distance=1)
    assert pos_dist == 1

    # 2. Negatives must NOT be within distance <= d_threshold
    # (i.e. distance in [0, d_threshold] is excluded)
    assert len(negs) == k_negatives
    for neg in negs:
        assert not is_within_distance(anchor, neg, max_distance=d_threshold)
        assert find_distance(anchor, neg, max_distance=d_threshold) is None


def test_generate_contrastive_batch_shapes():
    """Verify batch tensor shapes for anchors, positives, and negatives."""
    batch_size = 4
    k_negatives = 3
    anchor_t, pos_t, neg_t = generate_contrastive_batch(
        batch_size=batch_size,
        k_negatives=k_negatives,
        d_threshold=5,
    )

    assert anchor_t.shape == (batch_size, 54)
    assert pos_t.shape == (batch_size, 54)
    assert neg_t.shape == (batch_size, k_negatives, 54)
    assert anchor_t.dtype == torch.long
    assert pos_t.dtype == torch.long
    assert neg_t.dtype == torch.long


def test_contrastive_iterable_dataset():
    """Verify ContrastiveCubeIterableDataset streams expected batches."""
    steps = 2
    batch_size = 3
    dataset = ContrastiveCubeIterableDataset(
        total_steps=steps,
        batch_size=batch_size,
        k_negatives=2,
        d_threshold=5,
    )

    batches_seen = 0
    for a, p, n in dataset:
        assert a.shape == (batch_size, 54)
        assert p.shape == (batch_size, 54)
        assert n.shape == (batch_size, 2, 54)
        batches_seen += 1

    assert batches_seen == steps


def test_infonce_loss_math_and_gradients():
    """Verify InfoNCE loss calculation, accuracy, and gradient backpropagation."""
    criterion = InfoNCELoss(temperature=0.1)

    B, D, K = 2, 16, 4
    anchor = torch.randn(B, D, requires_grad=True)
    # Perfect positive (identical to anchor)
    pos_close = anchor.clone()
    # Distinct orthogonal negatives
    negs = torch.randn(B, K, D, requires_grad=True)

    loss_low, acc_high, metrics = criterion(anchor, pos_close, negs)

    assert loss_low.item() < 0.1
    assert acc_high.item() == 1.0
    assert metrics["margin"] > 0.5

    # Check gradient flow
    loss_low.backward()
    assert anchor.grad is not None
    assert negs.grad is not None
    assert not torch.isnan(anchor.grad).any()


def test_training_step_integration():
    """Verify a complete forward-backward-optimizer step executes without errors."""
    device = torch.device("cpu")
    model = CubeTransformerEncoder(
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        d_embed=32,
        dropout=0.0,
    ).to(device)

    criterion = InfoNCELoss(temperature=0.07).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    anchor_t, pos_t, neg_t = generate_contrastive_batch(
        batch_size=2,
        k_negatives=2,
        d_threshold=3,
        device=device,
    )

    B, K, _ = neg_t.shape
    anchor_emb, _ = model(anchor_t)
    pos_emb, _ = model(pos_t)

    neg_flat = neg_t.view(B * K, 54)
    neg_emb_flat, _ = model(neg_flat)
    neg_embs = neg_emb_flat.view(B, K, -1)

    loss, acc, metrics = criterion(anchor_emb, pos_emb, neg_embs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert 0.0 <= acc.item() <= 1.0


def test_concatenated_forward_pass_equivalence():
    """Verify that encoding all cubes in a single concatenated batch yields equivalent embeddings."""
    device = torch.device("cpu")
    model = CubeTransformerEncoder(
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        d_embed=32,
        dropout=0.0,
    ).to(device)
    model.eval()

    anchor_t, pos_t, neg_t = generate_contrastive_batch(
        batch_size=2,
        k_negatives=3,
        d_threshold=3,
        device=device,
    )

    B, K, _ = neg_t.shape
    neg_flat = neg_t.view(B * K, 54)

    # Separate forward passes
    with torch.no_grad():
        a_emb_sep, _ = model(anchor_t)
        p_emb_sep, _ = model(pos_t)
        n_emb_sep, _ = model(neg_flat)

        # Concatenated forward pass
        all_cubes = torch.cat([anchor_t, pos_t, neg_flat], dim=0)
        all_embs, _ = model(all_cubes)

        a_emb_cat = all_embs[:B]
        p_emb_cat = all_embs[B : 2 * B]
        n_emb_cat = all_embs[2 * B :]

    assert torch.allclose(a_emb_sep, a_emb_cat, atol=1e-5)
    assert torch.allclose(p_emb_sep, p_emb_cat, atol=1e-5)
    assert torch.allclose(n_emb_sep, n_emb_cat, atol=1e-5)


def test_interactive_train_infonce():
    """Verify train_infonce executes cleanly and returns TrainResult with valid history."""
    from rubik.trainer import TrainingConfig, train_infonce

    result = train_infonce(
        steps=3,
        batch_size=2,
        k_negatives=4,
        d_model=16,
        num_layers=1,
        nhead=2,
        d_embed=16,
        device="cpu",
        save_dir=None,
        show_progress_bar=False,
    )

    assert result.model is not None
    assert result.best_loss < float("inf")
    assert result.best_step >= 1
    assert len(result.history) > 0
    assert "loss" in result.history[0]
    assert "top1_acc" in result.history[0]
    assert "margin" in result.history[0]


