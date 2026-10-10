"""Pure Python functional, stateless representation of Rubik's cube using permutation lookups."""

import random
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")

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

# Precomputed permutation lookup tuples (54 indices each)
# new_state[i] = old_state[p[i]]
PERMUTATIONS: Dict[str, Tuple[int, ...]] = {
    "B": (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 12, 13, 14, 15, 16, 17, 27, 28, 29, 21,
        22, 23, 24, 25, 26, 36, 37, 38, 30, 31, 32, 33, 34, 35, 9, 10, 11, 39, 40, 41,
        42, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47,
    ),
    "B'": (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 36, 37, 38, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21,
        22, 23, 24, 25, 26, 18, 19, 20, 30, 31, 32, 33, 34, 35, 27, 28, 29, 39, 40, 41,
        42, 43, 44, 47, 50, 53, 46, 49, 52, 45, 48, 51,
    ),
    "D": (
        36, 37, 38, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 21,
        22, 23, 24, 25, 26, 33, 30, 27, 34, 31, 28, 35, 32, 29, 45, 46, 47, 39, 40, 41,
        42, 43, 44, 18, 19, 20, 48, 49, 50, 51, 52, 53,
    ),
    "D'": (
        18, 19, 20, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 45, 46, 47, 21,
        22, 23, 24, 25, 26, 29, 32, 35, 28, 31, 34, 27, 30, 33, 0, 1, 2, 39, 40, 41,
        42, 43, 44, 36, 37, 38, 48, 49, 50, 51, 52, 53,
    ),
    "F": (
        6, 3, 0, 7, 4, 1, 8, 5, 2, 36, 37, 38, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21,
        22, 23, 24, 25, 26, 18, 19, 20, 30, 31, 32, 33, 34, 35, 27, 28, 29, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    ),
    "F'": (
        2, 5, 8, 1, 4, 7, 0, 3, 6, 18, 19, 20, 12, 13, 14, 15, 16, 17, 27, 28, 29, 21,
        22, 23, 24, 25, 26, 36, 37, 38, 30, 31, 32, 33, 34, 35, 9, 10, 11, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    ),
    "L": (
        9, 10, 11, 3, 4, 5, 6, 7, 8, 45, 46, 47, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 0, 1, 2, 30, 31, 32, 33, 34, 35, 42, 39, 36, 43, 40, 37,
        44, 41, 38, 27, 28, 29, 48, 49, 50, 51, 52, 53,
    ),
    "L'": (
        27, 28, 29, 3, 4, 5, 6, 7, 8, 0, 1, 2, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 45, 46, 47, 30, 31, 32, 33, 34, 35, 38, 41, 44, 37, 40, 43,
        36, 39, 42, 9, 10, 11, 48, 49, 50, 51, 52, 53,
    ),
    "R": (
        27, 28, 29, 3, 4, 5, 6, 7, 8, 0, 1, 2, 12, 13, 14, 15, 16, 17, 24, 21, 18, 25,
        22, 19, 26, 23, 20, 45, 46, 47, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 9, 10, 11, 48, 49, 50, 51, 52, 53,
    ),
    "R'": (
        9, 10, 11, 3, 4, 5, 6, 7, 8, 45, 46, 47, 12, 13, 14, 15, 16, 17, 20, 23, 26, 19,
        22, 25, 18, 21, 24, 0, 1, 2, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 27, 28, 29, 48, 49, 50, 51, 52, 53,
    ),
    "U": (
        18, 19, 20, 3, 4, 5, 6, 7, 8, 15, 12, 9, 16, 13, 10, 17, 14, 11, 45, 46, 47, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 39, 40, 41,
        42, 43, 44, 36, 37, 38, 48, 49, 50, 51, 52, 53,
    ),
    "U'": (
        36, 37, 38, 3, 4, 5, 6, 7, 8, 11, 14, 17, 10, 13, 16, 9, 12, 15, 0, 1, 2, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 45, 46, 47, 39, 40, 41,
        42, 43, 44, 18, 19, 20, 48, 49, 50, 51, 52, 53,
    ),
}

# Compiled itemgetter functions executed in C inside CPython
GETTERS: Dict[str, Callable[[Any], Tuple[Any, ...]]] = {
    m: itemgetter(*p) for m, p in PERMUTATIONS.items()
}
GETTER_LIST: List[Callable[[Any], Tuple[Any, ...]]] = [GETTERS[m] for m in VALID_MOVES]
MOVE_TO_INDEX: Dict[str, int] = {m: i for i, m in enumerate(VALID_MOVES)}


def create_solved_cube() -> Tuple[str, ...]:
    """Return a 54-element tuple representing a solved Rubik's cube."""
    return tuple(SOLVED_FACELET)


def to_facelet_string(state: Sequence[Any]) -> str:
    """Convert a 54-element state sequence into a facelet string."""
    return "".join(str(x) for x in state)


def from_facelet_string(facelet: str) -> Tuple[str, ...]:
    """Convert a 54-character facelet string into an immutable tuple."""
    if len(facelet) != 54:
        raise ValueError(f"Facelet string must be 54 characters long, got {len(facelet)}")
    return tuple(facelet)


def apply_move(state: Sequence[T], move: Union[str, int]) -> Tuple[T, ...]:
    """
    Apply a single move functionally and return the updated state tuple.

    Args:
        state: Sequence (tuple, list, str) of length 54.
        move: Move name string (e.g., 'F', "U'") or move index 0..11.

    Returns:
        New 54-element tuple with rearranged positions.
    """
    getter = GETTER_LIST[move] if isinstance(move, int) else GETTERS[move]
    return getter(state)


def apply_moves(state: Sequence[T], moves: Sequence[Union[str, int]]) -> Tuple[T, ...]:
    """
    Apply a sequence of moves sequentially.

    Args:
        state: Sequence of length 54.
        moves: Sequence of move strings or integer indices.

    Returns:
        New 54-element tuple after applying all moves.
    """
    curr = state
    for m in moves:
        getter = GETTER_LIST[m] if isinstance(m, int) else GETTERS[m]
        curr = getter(curr)
    return curr if isinstance(curr, tuple) else tuple(curr)


def compose_moves(moves: Sequence[Union[str, int]]) -> Tuple[int, ...]:
    """
    Compose a sequence of moves into a single permutation tuple.

    For any state S: apply_moves(S, moves) == apply_move(S, compose_moves(moves)).

    Args:
        moves: Sequence of move strings or integer indices.

    Returns:
        54-element tuple representing the combined permutation indices.
    """
    p: Tuple[int, ...] = tuple(range(54))
    for m in moves:
        getter = GETTER_LIST[m] if isinstance(m, int) else GETTERS[m]
        p = getter(p)
    return p


def apply_move_batch(states: Sequence[Sequence[T]], move: Union[str, int]) -> List[Tuple[T, ...]]:
    """
    Apply a move across a batch of cube states.

    Args:
        states: Sequence of 54-element sequences.
        move: Move name or index.

    Returns:
        List of updated 54-element tuples.
    """
    getter = GETTER_LIST[move] if isinstance(move, int) else GETTERS[move]
    return [getter(s) for s in states]


def scramble(
    state: Optional[Sequence[T]] = None,
    n_moves: int = 20,
    avoid_repeated: bool = False,
    rng: Optional[Union[random.Random, Any]] = None,
) -> Tuple[Tuple[Any, ...], List[str]]:
    """
    Scramble the cube by applying n_moves randomly.

    Args:
        state: Initial cube state. Defaults to a solved cube tuple.
        n_moves: Number of random moves to perform.
        avoid_repeated: If True, avoid revisiting previously visited states during scramble.
        rng: Optional random.Random instance.

    Returns:
        (scrambled_state, solution_moves): Tuple with the final scrambled state tuple
        and the list of moves that solve the cube (in reverse order).
    """
    curr: Tuple[Any, ...] = tuple(state) if state is not None else create_solved_cube()
    rand_choice = rng.randrange if rng is not None else random.randrange
    solution: List[str] = []

    if not avoid_repeated:
        move_indices = [rand_choice(12) for _ in range(n_moves)]
        for idx in move_indices:
            curr = GETTER_LIST[idx](curr)
            move_name = VALID_MOVES[idx]
            solution.insert(0, INVERSE_MOVES[move_name])
        return curr, solution

    visited = {to_facelet_string(curr)}
    for _ in range(n_moves):
        idx = rand_choice(12)
        next_state = GETTER_LIST[idx](curr)
        next_facelet = to_facelet_string(next_state)

        while next_facelet in visited:
            idx = rand_choice(12)
            next_state = GETTER_LIST[idx](curr)
            next_facelet = to_facelet_string(next_state)

        curr = next_state
        visited.add(next_facelet)
        solution.insert(0, INVERSE_MOVES[VALID_MOVES[idx]])

    return curr, solution
