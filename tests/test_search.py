"""Unit tests for the neighborhood search and distance algorithms."""

from random import choices
import pytest

from rubik.cube import (
    SOLVED_FACELET,
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


def test_get_neighbors():
    """Verify get_neighbors returns all 12 valid moves with correct types."""
    solved = create_solved_cube()
    neighbors = get_neighbors(solved)
    assert len(neighbors) == 12
    assert set(neighbors.keys()) == set(VALID_MOVES)
    for m, s in neighbors.items():
        assert isinstance(s, tuple)
        assert len(s) == 54
        assert s != solved


def test_generate_neighborhood_tree():
    """Verify neighborhood tree generation on both unique IDs and color cubes."""
    # 1. On unique sticker IDs (no color symmetry collapse)
    unique_ids = tuple(range(54))
    tree_ids_0 = generate_neighborhood_tree(unique_ids, max_depth=0)
    assert len(tree_ids_0) == 1
    assert tree_ids_0[unique_ids] == 0

    tree_ids_1 = generate_neighborhood_tree(unique_ids, max_depth=1)
    assert len(tree_ids_1) == 13  # 1 root + 12 moves

    tree_ids_2 = generate_neighborhood_tree(unique_ids, max_depth=2)
    assert len(tree_ids_2) == 127  # 1 + 12 + 114 unique permutations

    tree_ids_3 = generate_neighborhood_tree(unique_ids, max_depth=3)
    assert len(tree_ids_3) == 1195

    # 2. On solved color cube (where identical sticker colors group into symmetric patterns)
    solved = create_solved_cube()
    tree_color_0 = generate_neighborhood_tree(solved, max_depth=0)
    assert len(tree_color_0) == 1

    tree_color_1 = generate_neighborhood_tree(solved, max_depth=1)
    assert len(tree_color_1) == 7

    tree_color_2 = generate_neighborhood_tree(solved, max_depth=2)
    assert len(tree_color_2) == 58

    tree_color_3 = generate_neighborhood_tree(solved, max_depth=3)
    assert len(tree_color_3) == 538


def test_distance_depth_zero():
    """Verify same state has distance 0 and is within distance."""
    solved = create_solved_cube()
    assert is_within_distance(solved, solved, max_distance=3)
    assert find_distance(solved, solved, max_distance=3) == 0
    assert find_path(solved, solved, max_distance=3) == []


@pytest.mark.parametrize("n_moves", [1, 2, 3])
def test_distance_within_three_moves(n_moves):
    """Verify states reachable in <= 3 moves are identified and path reconstructs target."""
    solved = create_solved_cube()
    moves = choices(VALID_MOVES, k=n_moves)
    target = apply_moves(solved, moves)

    assert is_within_distance(solved, target, max_distance=3)
    dist = find_distance(solved, target, max_distance=3)
    assert dist is not None
    assert dist <= n_moves

    path = find_path(solved, target, max_distance=3)
    assert path is not None
    assert len(path) == dist
    # Applying the found path must reconstruct target exactly
    assert apply_moves(solved, path) == target


def test_distance_exceeding_threshold():
    """Verify states scrambled with many random moves are detected as NOT within 3 moves."""
    solved = create_solved_cube()
    far_state, _ = scramble(solved, n_moves=20)

    # 20 random moves is practically guaranteed to be > 3 moves away
    assert not is_within_distance(solved, far_state, max_distance=3)
    assert find_distance(solved, far_state, max_distance=3) is None
    assert find_path(solved, far_state, max_distance=3) is None


def test_string_inputs():
    """Verify search functions accept facelet strings seamlessly."""
    solved_str = SOLVED_FACELET
    neighbors = get_neighbors(solved_str)
    assert len(neighbors) == 12

    # Target 2 moves away
    path_expected = ["F", "U"]
    target_tuple = apply_moves(create_solved_cube(), path_expected)
    target_str = to_facelet_string(target_tuple)

    assert is_within_distance(solved_str, target_str, max_distance=3)
    assert find_distance(solved_str, target_str, max_distance=3) == 2
    path = find_path(solved_str, target_str, max_distance=3)
    assert path is not None
    assert len(path) == 2
    assert apply_moves(solved_str, path) == target_tuple


def test_invalid_input_length():
    """Verify ValueError is raised for invalid length states."""
    with pytest.raises(ValueError):
        is_within_distance("short_string", SOLVED_FACELET)
