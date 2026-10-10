#!/usr/bin/env python3
"""Demonstration script: Neighborhood search and distance verification between scrambles."""

import time
from random import choices

from rubik.cube import (
    VALID_MOVES,
    apply_moves,
    create_solved_cube,
    scramble,
    to_facelet_string,
)
from rubik.search import (
    find_distance,
    find_path,
    generate_neighborhood_tree,
    get_neighbors,
    is_within_distance,
)


def main():
    print("=" * 76)
    print(" RUBIK CUBE NEIGHBORHOOD SEARCH & DISTANCE VERIFICATION (< 4 MOVES)")
    print("=" * 76)

    s0 = create_solved_cube()

    # 1. 1-Move Neighborhood
    neighbors = get_neighbors(s0)
    print(f"[1] 1-Move Neighbors of Solved Cube: {len(neighbors)} branching states")
    sample_moves = list(neighbors.keys())[:4]
    print(f"    Sample moves: {', '.join(sample_moves)} ...")

    # 2. Neighborhood Tree Generation
    print("\n[2] Expanding BFS Neighborhood Tree up to Depth 3:")
    for depth in range(4):
        t0 = time.perf_counter()
        tree = generate_neighborhood_tree(s0, max_depth=depth)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"    Depth {depth}: {len(tree):4d} unique states (expanded in {elapsed:6.2f} ms)")

    # 3. Verifying distance between pairs of scrambles
    print("\n[3] Testing Scramble Pairs (Distance Threshold: < 4 moves, i.e. <= 3 moves):")

    test_scrambles = [
        ("Identical States (0 moves)", []),
        ("1 Move Away", ["F"]),
        ("2 Moves Away", ["F", "U"]),
        ("3 Moves Away", ["F", "U'", "R"]),
        ("4 Moves Away", ["F", "U'", "R", "D"]),
        ("10 Moves Away", choices(VALID_MOVES, k=10)),
        ("20 Moves Away (Scramble)", None),
    ]

    for label, move_seq in test_scrambles:
        if move_seq is not None:
            target = apply_moves(s0, move_seq)
            actual_moves_applied = len(move_seq)
        else:
            target, solution = scramble(s0, n_moves=20)
            actual_moves_applied = 20

        t0 = time.perf_counter()
        within = is_within_distance(s0, target, max_distance=3)
        dist = find_distance(s0, target, max_distance=3)
        path = find_path(s0, target, max_distance=3)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        print(f"  * {label:28s} (Applied: {actual_moves_applied:2d} moves):")
        print(f"      Within < 4 moves?: {within} (Found dist: {dist})")
        if path is not None:
            print(f"      Shortest path    : {' '.join(path) if path else '(Identity)'}")
            assert apply_moves(s0, path) == target
        print(f"      Query time       : {elapsed_us:6.1f} µs")

    print("\n" + "=" * 76)
    print(" ALL VERIFICATIONS PASSED")
    print("=" * 76)


if __name__ == "__main__":
    main()
