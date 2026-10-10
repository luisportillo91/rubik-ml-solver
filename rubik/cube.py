"""Functional, stateless representation of Rubik's cube with precomputed permutation maps."""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

# Standard face order: F (0..8), U (9..17), R (18..26), D (27..35), L (36..44), B (45..53)
VALID_MOVES: Tuple[str, ...] = (
    "F", "F'", "U", "U'", "R", "R'", "D", "D'", "L", "L'", "B", "B'"
)

INVERSE_MOVES: Dict[str, str] = {
    "F": "F'",
    "F'": "F",
    "U": "U'",
    "U'": "U",
    "R": "R'",
    "R'": "R",
    "D": "D'",
    "D'": "D",
    "L": "L'",
    "L'": "L",
    "B": "B'",
    "B'": "B",
}

SOLVED_FACELET: str = "fffffffffuuuuuuuuurrrrrrrrrdddddddddlllllllllbbbbbbbbb"

# Precomputed permutation lookup maps (54 elements each)
# new_state[i] = old_state[p[i]]
PERMUTATIONS: Dict[str, np.ndarray] = {
    "B": np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 12, 13, 14, 15, 16, 17, 27, 28, 29, 21,
         22, 23, 24, 25, 26, 36, 37, 38, 30, 31, 32, 33, 34, 35, 9, 10, 11, 39, 40, 41,
         42, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47],
        dtype=np.int64,
    ),
    "B'": np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 36, 37, 38, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21,
         22, 23, 24, 25, 26, 18, 19, 20, 30, 31, 32, 33, 34, 35, 27, 28, 29, 39, 40, 41,
         42, 43, 44, 47, 50, 53, 46, 49, 52, 45, 48, 51],
        dtype=np.int64,
    ),
    "D": np.array(
        [36, 37, 38, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 21,
         22, 23, 24, 25, 26, 33, 30, 27, 34, 31, 28, 35, 32, 29, 45, 46, 47, 39, 40, 41,
         42, 43, 44, 18, 19, 20, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "D'": np.array(
        [18, 19, 20, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 45, 46, 47, 21,
         22, 23, 24, 25, 26, 29, 32, 35, 28, 31, 34, 27, 30, 33, 0, 1, 2, 39, 40, 41,
         42, 43, 44, 36, 37, 38, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "F": np.array(
        [6, 3, 0, 7, 4, 1, 8, 5, 2, 36, 37, 38, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21,
         22, 23, 24, 25, 26, 18, 19, 20, 30, 31, 32, 33, 34, 35, 27, 28, 29, 39, 40, 41,
         42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "F'": np.array(
        [2, 5, 8, 1, 4, 7, 0, 3, 6, 18, 19, 20, 12, 13, 14, 15, 16, 17, 27, 28, 29, 21,
         22, 23, 24, 25, 26, 36, 37, 38, 30, 31, 32, 33, 34, 35, 9, 10, 11, 39, 40, 41,
         42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "L": np.array(
        [9, 10, 11, 3, 4, 5, 6, 7, 8, 45, 46, 47, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
         22, 23, 24, 25, 26, 0, 1, 2, 30, 31, 32, 33, 34, 35, 42, 39, 36, 43, 40, 37,
         44, 41, 38, 27, 28, 29, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "L'": np.array(
        [27, 28, 29, 3, 4, 5, 6, 7, 8, 0, 1, 2, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
         22, 23, 24, 25, 26, 45, 46, 47, 30, 31, 32, 33, 34, 35, 38, 41, 44, 37, 40, 43,
         36, 39, 42, 9, 10, 11, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "R": np.array(
        [27, 28, 29, 3, 4, 5, 6, 7, 8, 0, 1, 2, 12, 13, 14, 15, 16, 17, 24, 21, 18, 25,
         22, 19, 26, 23, 20, 45, 46, 47, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
         42, 43, 44, 9, 10, 11, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "R'": np.array(
        [9, 10, 11, 3, 4, 5, 6, 7, 8, 45, 46, 47, 12, 13, 14, 15, 16, 17, 20, 23, 26, 19,
         22, 25, 18, 21, 24, 0, 1, 2, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
         42, 43, 44, 27, 28, 29, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "U": np.array(
        [18, 19, 20, 3, 4, 5, 6, 7, 8, 15, 12, 9, 16, 13, 10, 17, 14, 11, 45, 46, 47, 21,
         22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 39, 40, 41,
         42, 43, 44, 36, 37, 38, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
    "U'": np.array(
        [36, 37, 38, 3, 4, 5, 6, 7, 8, 11, 14, 17, 10, 13, 16, 9, 12, 15, 0, 1, 2, 21,
         22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 45, 46, 47, 39, 40, 41,
         42, 43, 44, 18, 19, 20, 48, 49, 50, 51, 52, 53],
        dtype=np.int64,
    ),
}

MOVE_ARRAY: np.ndarray = np.array([PERMUTATIONS[m] for m in VALID_MOVES], dtype=np.int64)
MOVE_TO_INDEX: Dict[str, int] = {m: i for i, m in enumerate(VALID_MOVES)}


def create_solved_cube(dtype=np.dtype("<U1")) -> np.ndarray:
    """Return a 1D NumPy array representing a solved Rubik's cube."""
    return np.array(list(SOLVED_FACELET), dtype=dtype)


def to_facelet_string(state: np.ndarray) -> str:
    """Convert a 54-element state array into a facelet string."""
    if state.dtype.kind in ("U", "S"):
        return "".join(state)
    return "".join(str(x) for x in state)


def from_facelet_string(facelet: str) -> np.ndarray:
    """Convert a 54-character facelet string into a 1D NumPy array."""
    if len(facelet) != 54:
        raise ValueError(f"Facelet string must be 54 characters long, got {len(facelet)}")
    return np.array(list(facelet), dtype="<U1")


def apply_move(state: np.ndarray, move: Union[str, int]) -> np.ndarray:
    """
    Apply a single move functionally and return the updated state array.

    Args:
        state: 1D NumPy array of length 54.
        move: Move name string (e.g., 'F', "U'") or move index 0..11.

    Returns:
        New 1D NumPy array with rearranged facelet positions.
    """
    perm = MOVE_ARRAY[move] if isinstance(move, int) else PERMUTATIONS[move]
    return state[perm]


def apply_moves(state: np.ndarray, moves: Sequence[Union[str, int]]) -> np.ndarray:
    """
    Apply a sequence of moves sequentially.

    Args:
        state: 1D NumPy array of length 54.
        moves: Sequence of move strings or integer indices.

    Returns:
        New 1D NumPy array after applying all moves.
    """
    curr = state
    for m in moves:
        curr = apply_move(curr, m)
    return curr


def compose_moves(moves: Sequence[Union[str, int]]) -> np.ndarray:
    """
    Compose a sequence of moves into a single permutation vector.

    For any state S: apply_moves(S, moves) == S[compose_moves(moves)].

    Args:
        moves: Sequence of move strings or integer indices.

    Returns:
        1D NumPy array of length 54 representing the combined permutation.
    """
    p = np.arange(54, dtype=np.int64)
    for m in moves:
        perm = MOVE_ARRAY[m] if isinstance(m, int) else PERMUTATIONS[m]
        p = p[perm]
    return p


def apply_move_batch(states: np.ndarray, move: Union[str, int]) -> np.ndarray:
    """
    Apply a move across a batch of cube states of shape (batch_size, 54).

    Args:
        states: 2D NumPy array of shape (B, 54).
        move: Move name or index.

    Returns:
        New 2D NumPy array of shape (B, 54).
    """
    perm = MOVE_ARRAY[move] if isinstance(move, int) else PERMUTATIONS[move]
    return states[:, perm]


def scramble(
    state: Optional[np.ndarray] = None,
    n_moves: int = 20,
    avoid_repeated: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Scramble the cube by applying n_moves randomly.

    Args:
        state: Initial cube state. Defaults to a solved cube.
        n_moves: Number of random moves to perform.
        avoid_repeated: If True, avoid revisiting previously visited states during scramble.
        rng: Optional NumPy random generator instance.

    Returns:
        (scrambled_state, solution_moves): Tuple with the final scrambled state
        and the list of moves that solve the cube (in reverse order).
    """
    if state is None:
        state = create_solved_cube()

    if rng is None:
        rng = np.random.default_rng()

    curr_state = state
    solution: List[str] = []

    if not avoid_repeated:
        move_indices = rng.integers(0, 12, size=n_moves)
        for idx in move_indices:
            curr_state = curr_state[MOVE_ARRAY[idx]]
            move_name = VALID_MOVES[idx]
            solution.insert(0, INVERSE_MOVES[move_name])
        return curr_state, solution

    visited = {to_facelet_string(curr_state)}
    for _ in range(n_moves):
        idx = int(rng.integers(0, 12))
        next_state = curr_state[MOVE_ARRAY[idx]]
        next_facelet = to_facelet_string(next_state)

        while next_facelet in visited:
            idx = int(rng.integers(0, 12))
            next_state = curr_state[MOVE_ARRAY[idx]]
            next_facelet = to_facelet_string(next_state)

        curr_state = next_state
        visited.add(next_facelet)
        solution.insert(0, INVERSE_MOVES[VALID_MOVES[idx]])

    return curr_state, solution
