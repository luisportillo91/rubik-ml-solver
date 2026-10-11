"""Unit tests for the 3-step windowed macro-solver."""

import pytest
import torch

from rubik.cube import apply_moves, create_solved_cube, scramble
from rubik.encoder import CubeTransformerEncoder
from rubik.solver import CubeSolver, SolvedBasin


@pytest.fixture(scope="module")
def small_model() -> CubeTransformerEncoder:
    """Fixture providing a lightweight CubeTransformerEncoder for fast testing."""
    model = CubeTransformerEncoder(
        d_model=64,
        nhead=2,
        num_layers=2,
        d_embed=64,
        dim_feedforward=128,
    )
    model.eval()
    return model


def test_solved_basin_properties():
    """Verify that SolvedBasin correctly precomputes exact shortest paths to solved."""
    basin = SolvedBasin(max_depth=3)
    assert len(basin.paths_to_solved) == 538
    solved = create_solved_cube()
    assert solved in basin.paths_to_solved
    assert basin.paths_to_solved[solved] == []

    # Verify that paths for random states in basin actually restore the solved cube
    for state, moves in list(basin.paths_to_solved.items())[:20]:
        end_state = apply_moves(state, moves)
        assert end_state == solved


def test_solver_already_solved(small_model):
    """Verify that an already solved cube immediately returns success with zero moves."""
    solver = CubeSolver(small_model, device="cpu", max_basin_depth=3)
    solved = create_solved_cube()
    result = solver.solve(solved)

    assert result.solved is True
    assert result.moves == []
    assert result.n_macro_steps == 0
    assert result.is_valid() is True


@pytest.mark.parametrize("scramble_dist", [1, 2, 3])
def test_solver_within_basin(small_model, scramble_dist):
    """Verify that scrambles within the basin radius (1-3 moves) are solved instantly with exact BFS."""
    solver = CubeSolver(small_model, device="cpu", max_basin_depth=3)
    solved = create_solved_cube()
    scrambled, orig_moves = scramble(solved, n_moves=scramble_dist)

    result = solver.solve(scrambled)
    assert result.solved is True
    assert result.is_valid() is True
    assert len(result.moves) <= scramble_dist


@pytest.mark.parametrize("scramble_dist", [4, 5, 6])
def test_solver_terminal_intersection(small_model, scramble_dist):
    """Verify that scrambles within radius 6 intersect the terminal basin via 3-move routes."""
    solver = CubeSolver(small_model, device="cpu", max_basin_depth=3)
    solved = create_solved_cube()
    scrambled, orig_moves = scramble(solved, n_moves=scramble_dist)

    result = solver.solve(scrambled, max_macro_steps=3)
    assert result.solved is True
    assert result.is_valid() is True


def test_solver_beam_search(small_model):
    """Verify that beam search executes without errors."""
    solver = CubeSolver(small_model, device="cpu", max_basin_depth=3)
    solved = create_solved_cube()
    scrambled, _ = scramble(solved, n_moves=4)

    result = solver.solve(scrambled, max_macro_steps=2, beam_width=2)
    assert result.solved is True
    assert result.is_valid() is True


def test_solver_target_modes(small_model):
    """Verify that both basin_max and solved target modes work properly."""
    for mode in ["basin_max", "solved", "basin_mean"]:
        solver = CubeSolver(small_model, device="cpu", max_basin_depth=2, target_mode=mode)
        solved = create_solved_cube()
        scrambled, _ = scramble(solved, n_moves=2)
        result = solver.solve(scrambled)
        assert result.solved is True
        assert result.is_valid() is True


def test_solve_scramble_helper(small_model):
    """Verify the solve_scramble helper method."""
    solver = CubeSolver(small_model, device="cpu", max_basin_depth=3)
    result, scramble_moves = solver.solve_scramble(n_moves=3)
    assert result.solved is True
    assert result.is_valid() is True
    assert len(scramble_moves) == 3
