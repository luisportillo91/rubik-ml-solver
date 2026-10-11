"""Neighborhood search and distance algorithms for Rubik's cube states."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from rubik.cube import GETTERS, apply_move


def _normalize_state(state: Union[str, Sequence[Any]]) -> Tuple[Any, ...]:
    if isinstance(state, str):
        if len(state) != 54:
            raise ValueError(f"State string must have length 54, got {len(state)}")
        return tuple(state)
    if isinstance(state, tuple):
        if len(state) != 54:
            raise ValueError(f"State tuple must have length 54, got {len(state)}")
        return state
    t = tuple(state)
    if len(t) != 54:
        raise ValueError(f"State sequence must have length 54, got {len(t)}")
    return t


def get_neighbors(state: Union[str, Sequence[Any]]) -> Dict[str, Tuple[Any, ...]]:
    """
    Return all 12 one-move neighbors of a state.

    Args:
        state: 54-element tuple or 54-character string.

    Returns:
        Dict mapping move name to the resulting neighbor state tuple.
    """
    norm = _normalize_state(state)
    return {m: getter(norm) for m, getter in GETTERS.items()}


def generate_neighborhood_tree(
    state: Union[str, Sequence[Any]], max_depth: int = 3
) -> Dict[Tuple[Any, ...], int]:
    """
    Generate all reachable states from the given state up to max_depth moves.

    Args:
        state: Initial state (tuple or string).
        max_depth: Maximum number of moves to explore (default 3).

    Returns:
        Dict mapping each reachable state tuple to its shortest move distance from state.
    """
    start = _normalize_state(state)
    visited: Dict[Tuple[Any, ...], int] = {start: 0}
    if max_depth <= 0:
        return visited

    frontier: List[Tuple[Any, ...]] = [start]
    for depth in range(1, max_depth + 1):
        next_frontier: List[Tuple[Any, ...]] = []
        for s in frontier:
            for getter in GETTERS.values():
                ns = getter(s)
                if ns not in visited:
                    visited[ns] = depth
                    next_frontier.append(ns)
        frontier = next_frontier

    return visited


def create_distance_filter(
    anchor: Union[str, Sequence[Any]], max_distance: int = 5
) -> Callable[[Union[str, Sequence[Any]]], bool]:
    """
    Create an optimized filter function to check whether candidate states
    are within max_distance of anchor.

    Precomputes the anchor's neighborhood tree once, eliminating repeated BFS
    searches across dozens or hundreds of candidates.

    Args:
        anchor: Anchor cube state.
        max_distance: Maximum distance threshold (default 5).

    Returns:
        Callable taking a candidate state and returning True if within max_distance.
    """
    s1 = _normalize_state(anchor)
    depth1 = (max_distance + 1) // 2
    depth2 = max_distance - depth1

    tree = generate_neighborhood_tree(s1, max_depth=depth1)

    def is_within(candidate: Union[str, Sequence[Any]]) -> bool:
        c = _normalize_state(candidate)
        if c in tree:
            return True
        if depth2 <= 0:
            return False

        d1 = []
        for getter in GETTERS.values():
            nc = getter(c)
            if nc in tree:
                return True
            d1.append(nc)

        if depth2 == 1:
            return False

        if depth2 == 2:
            for nc in d1:
                for getter in GETTERS.values():
                    if getter(nc) in tree:
                        return True
            return False

        return is_within_distance(s1, c, max_distance=max_distance)

    return is_within


def is_within_distance(
    state1: Union[str, Sequence[Any]],
    state2: Union[str, Sequence[Any]],
    max_distance: int = 3,
) -> bool:
    """
    Verify if state2 is reachable from state1 in at most max_distance moves.

    Args:
        state1: First cube state (e.g., initial scramble).
        state2: Second cube state (e.g., target scramble).
        max_distance: Maximum distance threshold (default 3, i.e., less than 4 moves).

    Returns:
        True if the distance between state1 and state2 is <= max_distance, else False.
    """
    return find_distance(state1, state2, max_distance=max_distance) is not None


def find_distance(
    state1: Union[str, Sequence[Any]],
    state2: Union[str, Sequence[Any]],
    max_distance: int = 3,
) -> Optional[int]:
    """
    Find the shortest move distance between state1 and state2 if <= max_distance.

    Uses bidirectional breadth-first search for optimal performance.

    Args:
        state1: Starting cube state.
        state2: Target cube state.
        max_distance: Maximum distance to search (default 3).

    Returns:
        Integer distance 0..max_distance if reachable, or None if not within range.
    """
    s1 = _normalize_state(state1)
    s2 = _normalize_state(state2)

    if s1 == s2:
        return 0
    if max_distance <= 0:
        return None

    # Bidirectional BFS
    fwd_visited: Dict[Tuple[Any, ...], int] = {s1: 0}
    fwd_frontier: List[Tuple[Any, ...]] = [s1]

    bwd_visited: Dict[Tuple[Any, ...], int] = {s2: 0}
    bwd_frontier: List[Tuple[Any, ...]] = [s2]

    fwd_depth = 0
    bwd_depth = 0

    while fwd_depth + bwd_depth < max_distance:
        if len(fwd_frontier) <= len(bwd_frontier):
            next_fwd: List[Tuple[Any, ...]] = []
            fwd_depth += 1
            for s in fwd_frontier:
                for getter in GETTERS.values():
                    ns = getter(s)
                    if ns in bwd_visited:
                        return fwd_depth + bwd_visited[ns]
                    if ns not in fwd_visited:
                        fwd_visited[ns] = fwd_depth
                        next_fwd.append(ns)
            fwd_frontier = next_fwd
            if not fwd_frontier:
                break
        else:
            next_bwd: List[Tuple[Any, ...]] = []
            bwd_depth += 1
            for s in bwd_frontier:
                for getter in GETTERS.values():
                    ns = getter(s)
                    if ns in fwd_visited:
                        return bwd_depth + fwd_visited[ns]
                    if ns not in bwd_visited:
                        bwd_visited[ns] = bwd_depth
                        next_bwd.append(ns)
            bwd_frontier = next_bwd
            if not bwd_frontier:
                break

    return None


def find_path(
    state1: Union[str, Sequence[Any]],
    state2: Union[str, Sequence[Any]],
    max_distance: int = 3,
) -> Optional[List[str]]:
    """
    Find the shortest sequence of moves transforming state1 into state2 if <= max_distance.

    Args:
        state1: Starting cube state.
        state2: Target cube state.
        max_distance: Maximum distance to search (default 3).

    Returns:
        List of move strings (e.g. ['F', "U'", 'R']) if reachable, or None if not within range.
    """
    s1 = _normalize_state(state1)
    s2 = _normalize_state(state2)

    if s1 == s2:
        return []
    if max_distance <= 0:
        return None

    # Parent pointers: state -> (parent_state, move_from_parent)
    fwd_parents: Dict[Tuple[Any, ...], Tuple[Optional[Tuple[Any, ...]], Optional[str]]] = {
        s1: (None, None)
    }
    fwd_frontier: List[Tuple[Any, ...]] = [s1]

    for d in range(1, max_distance + 1):
        next_fwd: List[Tuple[Any, ...]] = []
        for s in fwd_frontier:
            for move, getter in GETTERS.items():
                ns = getter(s)
                if ns not in fwd_parents:
                    fwd_parents[ns] = (s, move)
                    if ns == s2:
                        path: List[str] = []
                        curr: Optional[Tuple[Any, ...]] = ns
                        while curr is not None and fwd_parents[curr][0] is not None:
                            parent, m = fwd_parents[curr]
                            assert m is not None
                            path.append(m)
                            curr = parent
                        path.reverse()
                        return path
                    next_fwd.append(ns)
        fwd_frontier = next_fwd
        if not fwd_frontier:
            break

    return None
