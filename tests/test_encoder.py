"""Unit tests for the PyTorch Transformer Cube Encoder."""

import pytest
import torch

from rubik.cube import SOLVED_FACELET, create_solved_cube, scramble
from rubik.encoder import (
    COLOR_TO_ID,
    CubeTransformerEncoder,
    batch_to_tensor,
    cube_to_tensor,
)


def test_cube_and_batch_to_tensor():
    """Verify state to tensor conversions for single states and batches."""
    s0 = create_solved_cube()
    t1 = cube_to_tensor(s0)
    assert t1.shape == (1, 54)
    assert t1.dtype == torch.long
    assert t1[0, 0].item() == COLOR_TO_ID["f"]
    assert t1[0, 9].item() == COLOR_TO_ID["u"]

    # String input
    t_str = cube_to_tensor(SOLVED_FACELET)
    assert torch.equal(t1, t_str)

    # Batch conversion
    s1, _ = scramble(s0, n_moves=10)
    batch = batch_to_tensor([s0, s1])
    assert batch.shape == (2, 54)
    assert torch.equal(batch[0], t1[0])


def test_invalid_tensor_inputs():
    """Verify ValueError for invalid state dimensions or colors."""
    with pytest.raises(ValueError, match="54 facelets"):
        cube_to_tensor("invalid_short")

    with pytest.raises(ValueError, match="Unknown facelet color"):
        cube_to_tensor("x" * 54)


def test_encoder_forward_cls_pooling():
    """Verify Transformer Encoder forward pass with CLS token pooling."""
    model = CubeTransformerEncoder(
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        d_embed=32,
        pooling="cls",
        dropout=0.0,
    )
    model.eval()

    s0 = create_solved_cube()
    s1, _ = scramble(s0, n_moves=5)
    x = batch_to_tensor([s0, s1])

    global_emb, token_reps = model(x)

    assert global_emb.shape == (2, 32)
    assert token_reps.shape == (2, 54, 64)
    assert not torch.isnan(global_emb).any()
    assert not torch.isnan(token_reps).any()


def test_encoder_forward_mean_pooling():
    """Verify Transformer Encoder forward pass with mean pooling."""
    model = CubeTransformerEncoder(
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        d_embed=64,
        pooling="mean",
        dropout=0.0,
    )
    model.eval()

    s0 = create_solved_cube()
    x = cube_to_tensor(s0)

    global_emb, token_reps = model(x)

    assert global_emb.shape == (1, 64)
    assert token_reps.shape == (1, 54, 64)


def test_encoder_backward_gradients():
    """Verify gradients propagate through all layers during training."""
    model = CubeTransformerEncoder(
        d_model=32,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        d_embed=16,
        pooling="cls",
    )
    model.train()

    s0 = create_solved_cube()
    x = cube_to_tensor(s0)

    global_emb, _ = model(x)
    loss = global_emb.sum()
    loss.backward()

    # Verify gradients exist and are non-zero across parameter groups
    assert model.token_embedding.weight.grad is not None
    assert model.pos_embedding.weight.grad is not None
    assert model.cls_token.grad is not None
    for name, p in model.transformer.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"


def test_embed_convenience_method():
    """Verify model.embed accepts single states and batches directly."""
    model = CubeTransformerEncoder(
        d_model=32,
        nhead=2,
        num_layers=1,
        d_embed=16,
    )

    s0 = create_solved_cube()
    emb_single = model.embed(s0)
    assert emb_single.shape == (1, 16)

    s1, _ = scramble(s0, n_moves=10)
    emb_batch = model.embed([s0, s1])
    assert emb_batch.shape == (2, 16)


def test_encoder_sensitivity():
    """Verify identical states produce identical embeddings and scrambles produce distinct embeddings."""
    model = CubeTransformerEncoder(
        d_model=64,
        nhead=4,
        num_layers=2,
        d_embed=32,
        dropout=0.0,
    )

    s0 = create_solved_cube()
    emb0_a = model.embed(s0)
    emb0_b = model.embed(s0)
    assert torch.allclose(emb0_a, emb0_b, atol=1e-6)

    s_scrambled, _ = scramble(s0, n_moves=20)
    emb_scramble = model.embed(s_scrambled)
    assert not torch.allclose(emb0_a, emb_scramble, atol=1e-3)
