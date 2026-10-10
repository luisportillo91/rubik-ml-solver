#!/usr/bin/env python3
"""Validation script: verify scramble and inverse reconstruction consistency."""

import numpy as np
from random import choices

from rubik.cube import (
    INVERSE_MOVES,
    SOLVED_FACELET,
    VALID_MOVES,
    apply_moves,
    create_solved_cube,
    to_facelet_string,
)


def validate_scramble_reconstruction(n_moves: int, verbose: bool = True):
    # Functional representation with 54 distinct sticker IDs (0..53)
    sticker_ids = np.arange(54, dtype=np.int64)
    solved_colors = create_solved_cube()

    # Generate random sequence of moves
    scramble_moves = choices(VALID_MOVES, k=n_moves)
    # Conjugate inverse moves in reverse order: (M_1 M_2 ... M_k)^-1 = M_k^-1 ... M_2^-1 M_1^-1
    inverse_moves = [INVERSE_MOVES[m] for m in reversed(scramble_moves)]

    # Apply scramble moves
    scrambled_ids = apply_moves(sticker_ids, scramble_moves)
    scrambled_colors = apply_moves(solved_colors, scramble_moves)

    # Apply inverse moves in reverse order
    reconstructed_ids = apply_moves(scrambled_ids, inverse_moves)
    reconstructed_colors = apply_moves(scrambled_colors, inverse_moves)

    ids_match = np.array_equal(sticker_ids, reconstructed_ids)
    colors_match = np.array_equal(solved_colors, reconstructed_colors)
    facelet_match = (to_facelet_string(reconstructed_colors) == SOLVED_FACELET)

    if verbose:
        print(f"--- Test Scramble with {n_moves:4d} Moves ---")
        if n_moves <= 10:
            print(f"Scramble moves : {' '.join(scramble_moves)}")
            print(f"Inverse moves  : {' '.join(inverse_moves)}")
        else:
            print(f"Scramble prefix: {' '.join(scramble_moves[:5])} ... {' '.join(scramble_moves[-5:])}")
            print(f"Inverse prefix : {' '.join(inverse_moves[:5])} ... {' '.join(inverse_moves[-5:])}")

        print(f"  [✓] 54 Unique Sticker IDs exact match   : {ids_match}")
        print(f"  [✓] Facelet Colors exact match          : {colors_match}")
        print(f"  [✓] Facelet string matches SOLVED_STATE : {facelet_match}")
        print()

    assert ids_match, f"Sticker IDs did not match for {n_moves} moves!"
    assert colors_match, f"Colors did not match for {n_moves} moves!"
    assert facelet_match, f"Facelet did not match for {n_moves} moves!"


def main():
    print("=" * 70)
    print(" VALIDATING CUBE CONSISTENCY (SCRAMBLE + INVERSE RECONSTRUCTION)")
    print("=" * 70)

    test_lengths = [5, 10, 50, 100, 500, 1000]
    for n in test_lengths:
        validate_scramble_reconstruction(n, verbose=True)

    print("=" * 70)
    print(" ALL CONSISTENCY CHECKS PASSED: Reconstruction is 100% lossless!")
    print("=" * 70)


if __name__ == "__main__":
    main()
