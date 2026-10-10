"""Unit tests for the pure Python functional Rubik's cube representation."""

from operator import itemgetter
from random import choices
import pytest

from rubik.cube import (
    GETTERS,
    GETTER_LIST,
    INVERSE_MOVES,
    PERMUTATIONS,
    SOLVED_FACELET,
    VALID_MOVES,
    apply_move,
    apply_move_batch,
    apply_moves,
    compose_moves,
    create_solved_cube,
    from_facelet_string,
    scramble,
    to_facelet_string,
)


def test_permutations_are_valid_bijections():
    """Verify each permutation is a bijection over indices 0..53."""
    expected = list(range(54))
    for move in VALID_MOVES:
        perm = PERMUTATIONS[move]
        assert len(perm) == 54
        assert len(set(perm)) == 54
        assert sorted(perm) == expected


def test_inverses():
    """Verify that every move followed by its inverse yields identity."""
    identity = tuple(range(54))
    for move in VALID_MOVES:
        inv_move = INVERSE_MOVES[move]
        res = apply_move(apply_move(identity, move), inv_move)
        assert res == identity


def test_apply_moves_and_composition():
    """Verify that composing moves gives the exact same result as sequential application."""
    moves = choices(VALID_MOVES, k=25)
    state = create_solved_cube()

    seq_state = apply_moves(state, moves)
    composed_perm = compose_moves(moves)
    composed_state = itemgetter(*composed_perm)(state)

    assert seq_state == composed_state


def test_facelet_string_roundtrip():
    """Verify from_facelet_string and to_facelet_string are lossless inverses."""
    state = create_solved_cube()
    s = to_facelet_string(state)
    assert s == SOLVED_FACELET
    assert from_facelet_string(s) == state


def test_scramble_roundtrip():
    """Verify that scrambling and applying the returned solution solves the cube."""
    state, solution = scramble(n_moves=30)
    assert to_facelet_string(state) != SOLVED_FACELET

    solved = apply_moves(state, solution)
    assert to_facelet_string(solved) == SOLVED_FACELET


def test_scramble_avoid_repeated():
    """Verify scramble with avoid_repeated=True generates valid states."""
    state, solution = scramble(n_moves=15, avoid_repeated=True)
    solved = apply_moves(state, solution)
    assert to_facelet_string(solved) == SOLVED_FACELET


def test_apply_move_batch():
    """Verify batch move execution on a list of state tuples."""
    batch_size = 10
    base_state = create_solved_cube()
    batch = [base_state for _ in range(batch_size)]

    for move in VALID_MOVES:
        batch_res = apply_move_batch(batch, move)
        single_res = apply_move(base_state, move)
        assert len(batch_res) == batch_size
        for item in batch_res:
            assert item == single_res


@pytest.mark.parametrize("n_moves", [10, 50, 100, 500, 1000])
def test_conjugate_inverse_reconstruction(n_moves):
    """Verify that applying inverse movements in reverse order reconstructs the exact initial state."""
    # Test on unique sticker IDs (0..53) to verify zero permutation drift
    initial_ids = tuple(range(54))
    initial_colors = create_solved_cube()

    scramble_moves = choices(VALID_MOVES, k=n_moves)
    inverse_moves = [INVERSE_MOVES[m] for m in reversed(scramble_moves)]

    reconstructed_ids = apply_moves(apply_moves(initial_ids, scramble_moves), inverse_moves)
    reconstructed_colors = apply_moves(apply_moves(initial_colors, scramble_moves), inverse_moves)

    assert reconstructed_ids == initial_ids
    assert reconstructed_colors == initial_colors
    assert to_facelet_string(reconstructed_colors) == SOLVED_FACELET
