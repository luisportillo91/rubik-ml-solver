"""On-the-fly streaming generator for Rubik's cube contrastive learning."""

import random
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch.utils.data import IterableDataset

from rubik.cube import (
    PERMUTATIONS,
    VALID_MOVES,
    apply_move,
    compose_moves,
    create_solved_cube,
    scramble,
)
from rubik.encoder import _state_to_ids, batch_to_tensor
from rubik.search import (
    create_distance_filter,
    generate_neighborhood_tree,
    is_within_distance,
)


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
        - Negatives: K states with distance > d_threshold generated via random walk from hashmap.

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
    # Sequential walk from anchor:
    # Steps 2..d_threshold are in excluded buffer zone.
    # Steps > d_threshold: each movement step along the walk is a negative state!
    is_within = create_distance_filter(anchor, max_distance=d_threshold)
    negatives: List[Tuple[str, ...]] = []
    curr = anchor
    step = 0
    while len(negatives) < k_negatives:
        curr = apply_move(curr, rand.choice(VALID_MOVES))
        step += 1
        if step > d_threshold:
            if not is_within(curr):
                negatives.append(curr)

    return anchor, positive, negatives


def generate_contrastive_batch(
    batch_size: int = 32,
    k_negatives: int = 4,
    d_threshold: int = 5,
    min_scramble: int = 1,
    max_scramble: int = 25,
    n_trajectories: int = 1,
    device: Optional[torch.device] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of contrastive pairs directly as PyTorch tensors.

    Args:
        batch_size: Number of anchor states in the batch (B).
        k_negatives: Total number of negative movement steps per anchor (K).
        d_threshold: Distance threshold d for excluded buffer zone [2, d_threshold] (default 5).
        min_scramble: Minimum moves for scramble generation.
        max_scramble: Maximum moves for scramble generation.
        n_trajectories: Number of canonical trajectories to distribute negative movements across (default 1).
        device: Target torch device.
        rng: Optional random.Random instance.

    Returns:
        Tuple of:
            - anchor_tensor: (B, 54) LongTensor
            - pos_tensor: (B, 54) LongTensor
            - neg_tensor: (B, K, 54) LongTensor
    """
    rand = rng if rng is not None else random
    solved = create_solved_cube()

    # 1. Generate B random anchor states
    anchors_ids: List[List[int]] = []
    for _ in range(batch_size):
        n_moves = rand.randint(min_scramble, max_scramble)
        anchor, _ = scramble(solved, n_moves=n_moves, rng=rand)
        anchors_ids.append(_state_to_ids(anchor))

    anchor_tensor = torch.tensor(anchors_ids, dtype=torch.long, device=device)

    # 2. Pick a random 1-step move for each anchor (vectorized)
    move_names = list(VALID_MOVES)
    pos_perms = [PERMUTATIONS[rand.choice(move_names)] for _ in range(batch_size)]
    pos_perms_t = torch.tensor(pos_perms, dtype=torch.long, device=device)
    pos_tensor = anchor_tensor.gather(1, pos_perms_t)

    # 3. Canonical sequential trajectories:
    # A canonical trajectory is a sequence of movements.
    # Movement 1: positive, Movements 2..d_threshold: excluded buffer zone.
    # Subsequent movements (> d_threshold): each movement produces a negative permutation.
    neg_perms: List[Tuple[int, ...]] = []
    filter_fn = create_distance_filter(solved, max_distance=d_threshold)
    n_trajs = max(1, n_trajectories)
    moves_per_traj = (k_negatives + n_trajs - 1) // n_trajs

    for _ in range(n_trajs):
        curr = solved
        moves_history: List[str] = []
        step = 0
        traj_negs = 0
        while traj_negs < moves_per_traj and len(neg_perms) < k_negatives:
            m = rand.choice(VALID_MOVES)
            moves_history.append(m)
            curr = apply_move(curr, m)
            step += 1
            if step > d_threshold:
                if step < 16:
                    if not filter_fn(curr):
                        neg_perms.append(compose_moves(moves_history))
                        traj_negs += 1
                else:
                    neg_perms.append(compose_moves(moves_history))
                    traj_negs += 1

    neg_perms_t = torch.tensor(neg_perms, dtype=torch.long, device=device)

    # 4. Vectorized PyTorch gather across all B anchors and K negatives
    neg_tensor = anchor_tensor.unsqueeze(1).expand(-1, k_negatives, -1).gather(
        2, neg_perms_t.unsqueeze(0).expand(batch_size, -1, -1)
    )

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
        n_trajectories: int = 1,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.k_negatives = k_negatives
        self.d_threshold = d_threshold
        self.min_scramble = min_scramble
        self.max_scramble = max_scramble
        self.n_trajectories = n_trajectories

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        step = 0
        while self.total_steps is None or step < self.total_steps:
            yield generate_contrastive_batch(
                batch_size=self.batch_size,
                k_negatives=self.k_negatives,
                d_threshold=self.d_threshold,
                min_scramble=self.min_scramble,
                max_scramble=self.max_scramble,
                n_trajectories=self.n_trajectories,
            )
            step += 1

    def __len__(self) -> int:
        if self.total_steps is None:
            raise TypeError("Infinite streaming dataset has no predefined length.")
        return self.total_steps
