#!/usr/bin/env python3
"""Demonstration script: PyTorch Transformer Cube Encoder."""

import time
import torch
import torch.nn.functional as F

from rubik.cube import apply_move, create_solved_cube, scramble
from rubik.encoder import CubeTransformerEncoder, batch_to_tensor


def main():
    print("=" * 76)
    print(" PYTORCH TRANSFORMER CUBE ENCODER DEMO")
    print("=" * 76)

    # 1. Instantiate the Encoder
    d_model = 128
    d_embed = 128
    nhead = 4
    num_layers = 4
    encoder = CubeTransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=256,
        d_embed=d_embed,
        pooling="cls",
        dropout=0.0,
    )
    encoder.eval()

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"[1] Model Architecture:")
    print(f"    d_model: {d_model}, nhead: {nhead}, layers: {num_layers}, d_embed: {d_embed}")
    print(f"    Total Trainable Parameters: {num_params:,}")

    # 2. Encode Sample States
    s0 = create_solved_cube()
    s_1move = apply_move(s0, "F")
    s_2moves = apply_move(s_1move, "U")
    s_scramble_10, _ = scramble(s0, n_moves=10)
    s_scramble_20, _ = scramble(s0, n_moves=20)

    states = [s0, s_1move, s_2moves, s_scramble_10, s_scramble_20]
    labels = ["Solved", "1-Move (F)", "2-Moves (F U)", "10-Moves Scramble", "20-Moves Scramble"]

    # 3. Batch Forward Pass & Latency
    t0 = time.perf_counter()
    with torch.no_grad():
        x = batch_to_tensor(states)
        global_embs, token_reps = encoder(x)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[2] Forward Pass:")
    print(f"    Batch Size: {len(states)}")
    print(f"    Input Tensor Shape : {tuple(x.shape)}")
    print(f"    Global Embedding   : {tuple(global_embs.shape)}")
    print(f"    Token Representations: {tuple(token_reps.shape)}")
    print(f"    Inference Latency  : {latency_ms:.2f} ms ({latency_ms / len(states):.2f} ms/cube)")

    # 4. Cosine Similarities with Solved State
    print(f"\n[3] Cosine Similarity to Solved Cube Embedding:")
    solved_emb = global_embs[0:1]
    for i, label in enumerate(labels):
        sim = F.cosine_similarity(solved_emb, global_embs[i : i + 1]).item()
        l2_dist = torch.norm(solved_emb - global_embs[i : i + 1]).item()
        print(f"    {label:20s} -> Cosine Sim: {sim:+.4f} | L2 Dist: {l2_dist:.4f}")

    print("\n" + "=" * 76)
    print(" ENCODER DEMO COMPLETED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    main()
