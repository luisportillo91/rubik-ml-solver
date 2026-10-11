"""On-the-fly streaming generator for Rubik's cube contrastive learning."""

import random
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch.utils.data import IterableDataset

from rubik.cube import (
    VALID_MOVES,
    apply_move,
    create_solved_cube,
    scramble,
)
from rubik.encoder import _state_to_ids, batch_to_tensor
from rubik.search import is_within_distance


def generate_contrastive_sample(
    d_threshold: int = 5,
    k_negatives: int = 4,
    min_scramble: int = 1,
    max_scramble: int = 25,
    rng: Optional[random.Random] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], List[Tuple[str, ...]]]:
    """
    Generate a single (anchor, positive, negatives) triplet on-the-fly.

    Topology rules:
        - Anchor: random scramble.
        - Positive: exactly 1 step ahead (dist == 1).
        - Excluded buffer: any state with distance in [2, d_threshold] is excluded (default k=5).
        - Negatives: K states with distance > d_threshold.

    Args:
        d_threshold: Upper distance bound for excluded buffer zone (default 5).
        k_negatives: Number of negative states to sample (default 4).
        min_scramble: Minimum scramble length for anchor.
        max_scramble: Maximum scramble length for anchor.
        rng: Optional random.Random instance.

    Returns:
        (anchor_state, positive_state, list_of_negative_states)
    """
    rand = rng if rng is not None else random
    solved = create_solved_cube()

    # 1. Generate anchor state
    anchor_moves = rand.randint(min_scramble, max_scramble)
    anchor, _ = scramble(solved, n_moves=anchor_moves, rng=rand)

    # 2. Positive state: exactly 1 move ahead from anchor (dist == 1)
    move = rand.choice(VALID_MOVES)
    positive = apply_move(anchor, move)

    # 3. Sample K negatives strictly with distance > d_threshold
    negatives: List[Tuple[str, ...]] = []
    while len(negatives) < k_negatives:
        neg_moves = rand.randint(max(d_threshold + 5, 14), max_scramble + 5)
        candidate, _ = scramble(solved, n_moves=neg_moves, rng=rand)

        # Exclude candidates within distance <= d_threshold
        if not is_within_distance(anchor, candidate, max_distance=d_threshold):
            negatives.append(candidate)

    return anchor, positive, negatives


def generate_contrastive_batch(
    batch_size: int = 32,
    k_negatives: int = 4,
    d_threshold: int = 5,
    min_scramble: int = 1,
    max_scramble: int = 25,
    device: Optional[torch.device] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of contrastive pairs directly as PyTorch tensors.

    Args:
        batch_size: Number of anchor states in the batch (B).
        k_negatives: Number of negatives per anchor (K).
        d_threshold: Distance threshold d for negative sampling (default 5).
        min_scramble: Minimum moves for scramble generation.
        max_scramble: Maximum moves for scramble generation.
        device: Target torch device.
        rng: Optional random.Random instance.

    Returns:
        Tuple of:
            - anchor_tensor: (B, 54) LongTensor
            - pos_tensor: (B, 54) LongTensor
            - neg_tensor: (B, K, 54) LongTensor
    """
    anchors_ids: List[List[int]] = []
    pos_ids: List[List[int]] = []
    negs_ids: List[List[List[int]]] = []

    for _ in range(batch_size):
        anchor, pos, negs = generate_contrastive_sample(
            d_threshold=d_threshold,
            k_negatives=k_negatives,
            min_scramble=min_scramble,
            max_scramble=max_scramble,
            rng=rng,
        )
        anchors_ids.append(_state_to_ids(anchor))
        pos_ids.append(_state_to_ids(pos))
        negs_ids.append([_state_to_ids(n) for n in negs])

    anchor_tensor = torch.tensor(anchors_ids, dtype=torch.long, device=device)
    pos_tensor = torch.tensor(pos_ids, dtype=torch.long, device=device)
    neg_tensor = torch.tensor(negs_ids, dtype=torch.long, device=device)

    return anchor_tensor, pos_tensor, neg_tensor


class ContrastiveCubeIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset streaming infinite contrastive batches on-the-fly.

    Zero file storage needed. Synthesizes training data directly in memory.
    """

    def __init__(
        self,
        total_steps: Optional[int] = None,
        batch_size: int = 32,
        k_negatives: int = 4,
        d_threshold: int = 5,
        min_scramble: int = 1,
        max_scramble: int = 25,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.k_negatives = k_negatives
        self.d_threshold = d_threshold
        self.min_scramble = min_scramble
        self.max_scramble = max_scramble

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        step = 0
        while self.total_steps is None or step < self.total_steps:
            yield generate_contrastive_batch(
                batch_size=self.batch_size,
                k_negatives=self.k_negatives,
                d_threshold=self.d_threshold,
                min_scramble=self.min_scramble,
                max_scramble=self.max_scramble,
            )
            step += 1

    def __len__(self) -> int:
        if self.total_steps is None:
            raise TypeError("Infinite streaming dataset has no predefined length.")
        return self.total_steps
