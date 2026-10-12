"""3-Step Windowed Macro-Solver for Rubik's Cube Representation Learning."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F

from rubik.cube import (
    GETTERS,
    INVERSE_MOVES,
    PERMUTATIONS,
    VALID_MOVES,
    apply_move,
    apply_moves,
    compose_moves,
    create_solved_cube,
    scramble,
)
from rubik.encoder import CubeTransformerEncoder, _state_to_ids


CANONICAL_COMMUTE_ORDER: Dict[Tuple[str, str], bool] = {
    ("D", "U"): False,
    ("D'", "U"): False,
    ("D", "U'"): False,
    ("D'", "U'"): False,
    ("B", "F"): False,
    ("B'", "F"): False,
    ("B", "F'"): False,
    ("B'", "F'"): False,
    ("R", "L"): False,
    ("R'", "L"): False,
    ("R", "L'"): False,
    ("R'", "L'"): False,
}


@dataclass
class SolveResult:
    """Detailed result of a Rubik's cube solve attempt."""

    solved: bool
    moves: List[str]
    n_macro_steps: int
    duration_seconds: float
    initial_state: Tuple[str, ...]
    final_state: Tuple[str, ...]
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def move_count(self) -> int:
        """Total number of atomic face turns in the solution."""
        return len(self.moves)

    def is_valid(self) -> bool:
        """Verify whether applying the move sequence transforms initial state to solved."""
        solved_cube = create_solved_cube()
        curr = self.initial_state
        for m in self.moves:
            curr = apply_move(curr, m)
        return curr == solved_cube

    def __repr__(self) -> str:
        status = "SOLVED" if self.solved else "FAILED"
        return (
            f"SolveResult({status}, moves={self.move_count}, "
            f"macro_steps={self.n_macro_steps}, time={self.duration_seconds:.3f}s)"
        )


class SolvedBasin:
    """
    Precomputed terminal basin around the solved cube up to max_depth moves.

    Contains exact shortest paths to solved and precomputed embeddings for
    zero-error snapping within the basin radius.
    """

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.solved = create_solved_cube()
        # Map: state tuple -> list of inverse moves directly to solved
        self.paths_to_solved: Dict[Tuple[str, ...], List[str]] = {self.solved: []}

        frontier: List[Tuple[str, ...]] = [self.solved]
        for depth in range(1, max_depth + 1):
            next_frontier: List[Tuple[str, ...]] = []
            for s in frontier:
                for m, getter in GETTERS.items():
                    ns = getter(s)
                    if ns not in self.paths_to_solved:
                        inv_m = INVERSE_MOVES[m]
                        self.paths_to_solved[ns] = [inv_m] + self.paths_to_solved[s]
                        next_frontier.append(ns)
            frontier = next_frontier

        self.states: List[Tuple[str, ...]] = list(self.paths_to_solved.keys())
        self.tensor_ids = torch.tensor(
            [_state_to_ids(s) for s in self.states], dtype=torch.long
        )


class CubeSolver:
    """
    3-Step Windowed Macro-Solver for Rubik's Cube.

    Overcomes single-step representation noise by averaging cosine similarities
    across 3 consecutive movements, and snaps to exact solutions in the radius-6
    terminal intersection basin.
    """

    def __init__(
        self,
        model: CubeTransformerEncoder,
        device: Optional[Union[str, torch.device]] = None,
        max_basin_depth: int = 3,
        target_mode: str = "basin_max",
    ):
        """
        Initialize the CubeSolver.

        Args:
            model: Trained CubeTransformerEncoder instance.
            device: Target torch device ('auto', 'cuda', 'cpu', etc.).
            max_basin_depth: Radius of the exact terminal BFS basin (default 3).
            target_mode: Similarity objective:
                - 'basin_max': Max cosine similarity to any state in terminal basin (default).
                - 'solved': Direct cosine similarity to the solved cube embedding.
                - 'basin_mean': Average cosine similarity to the terminal basin.
        """
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.model.eval()

        self.target_mode = target_mode
        self.max_basin_depth = max_basin_depth
        self.basin = SolvedBasin(max_depth=max_basin_depth)

        # 1. Precompute target basin embeddings
        with torch.no_grad():
            basin_batch = self.basin.tensor_ids.to(self.device)
            embs, _ = self.model(basin_batch)
            self.basin_embs = F.normalize(embs, p=2, dim=-1)

            solved_ids = torch.tensor(
                _state_to_ids(self.basin.solved), dtype=torch.long, device=self.device
            ).unsqueeze(0)
            s_emb, _ = self.model(solved_ids)
            self.solved_emb = F.normalize(s_emb, p=2, dim=-1)

        # 2. Precompute non-redundant 3-move paths & permutation tensors
        self.paths: List[Tuple[str, str, str]] = self._generate_pruned_3move_paths()
        p1_perms = [compose_moves([p[0]]) for p in self.paths]
        p2_perms = [compose_moves([p[0], p[1]]) for p in self.paths]
        p3_perms = [compose_moves([p[0], p[1], p[2]]) for p in self.paths]

        # Deduplicate unique permutations across depths (1,194 instead of 3,528)
        self.u1 = list(dict.fromkeys(p1_perms))
        self.u2 = list(dict.fromkeys(p2_perms))
        self.u3 = list(dict.fromkeys(p3_perms))

        map1 = {p: i for i, p in enumerate(self.u1)}
        map2 = {p: i for i, p in enumerate(self.u2)}
        map3 = {p: i for i, p in enumerate(self.u3)}

        self.idx1 = torch.tensor([map1[p] for p in p1_perms], dtype=torch.long, device=self.device)
        self.idx2 = torch.tensor([map2[p] for p in p2_perms], dtype=torch.long, device=self.device)
        self.idx3 = torch.tensor([map3[p] for p in p3_perms], dtype=torch.long, device=self.device)

        self.len_u1 = len(self.u1)
        self.len_u2 = len(self.u2)

        all_perms = self.u1 + self.u2 + self.u3
        self.all_perms = torch.tensor(all_perms, dtype=torch.long, device=self.device)

    def _generate_pruned_3move_paths(self) -> List[Tuple[str, str, str]]:
        """Generate all non-redundant, pruned sequences of 3 consecutive moves."""
        paths = []
        for m1 in VALID_MOVES:
            for m2 in VALID_MOVES:
                if m2 == INVERSE_MOVES[m1] or (m1, m2) in CANONICAL_COMMUTE_ORDER:
                    continue
                for m3 in VALID_MOVES:
                    if (
                        m3 == INVERSE_MOVES[m2]
                        or (m2, m3) in CANONICAL_COMMUTE_ORDER
                        or m1 == m2 == m3
                    ):
                        continue
                    paths.append((m1, m2, m3))
        return paths

    def solve(
        self,
        state: Union[str, Sequence[Any]],
        max_macro_steps: int = 20,
        beam_width: int = 1,
    ) -> SolveResult:
        """
        Solve a scrambled Rubik's cube using 3-step windowed macro search.

        Args:
            state: Initial scrambled cube state (54-element tuple or string).
            max_macro_steps: Maximum number of 3-move macro steps to attempt.
            beam_width: Beam width (1 for greedy, >1 for beam search).

        Returns:
            SolveResult with success status, sequence of moves, and metrics.
        """
        t0 = time.perf_counter()
        solved_cube = self.basin.solved
        initial_state = tuple(state)

        # 0. Check if already solved
        if initial_state == solved_cube:
            return SolveResult(
                solved=True,
                moves=[],
                n_macro_steps=0,
                duration_seconds=time.perf_counter() - t0,
                initial_state=initial_state,
                final_state=initial_state,
            )

        # 1. Check if directly inside the precomputed basin (<= max_basin_depth)
        if initial_state in self.basin.paths_to_solved:
            moves = self.basin.paths_to_solved[initial_state]
            final_cube = apply_moves(initial_state, moves)
            return SolveResult(
                solved=True,
                moves=moves,
                n_macro_steps=0,
                duration_seconds=time.perf_counter() - t0,
                initial_state=initial_state,
                final_state=final_cube,
            )

        # 2. Macro Search Loop
        # A beam entry is: (cumulative_score, current_state, move_history, visited_states_in_path)
        beam: List[Tuple[float, Tuple[str, ...], List[str], set]] = [
            (0.0, initial_state, [], {initial_state})
        ]
        history: List[Dict[str, Any]] = []

        for step in range(1, max_macro_steps + 1):
            candidates: List[Tuple[float, Tuple[str, ...], List[str], set]] = []

            for cum_score, curr_state, move_history, visited in beam:
                last_move = move_history[-1] if move_history else None

                # Fast early intersection check: explore 1, 2, 3 moves
                # Depth 1 check (12 moves)
                for m1 in VALID_MOVES:
                    if last_move and m1 == INVERSE_MOVES[last_move]:
                        continue
                    s1 = GETTERS[m1](curr_state)
                    if s1 in self.basin.paths_to_solved:
                        full_moves = move_history + [m1] + self.basin.paths_to_solved[s1]
                        return SolveResult(
                            solved=True,
                            moves=full_moves,
                            n_macro_steps=step,
                            duration_seconds=time.perf_counter() - t0,
                            initial_state=initial_state,
                            final_state=apply_moves(curr_state, [m1] + self.basin.paths_to_solved[s1]),
                            history=history,
                        )

                # Depth 2 check (~114 moves)
                for m1 in VALID_MOVES:
                    if last_move and m1 == INVERSE_MOVES[last_move]:
                        continue
                    s1 = GETTERS[m1](curr_state)
                    for m2 in VALID_MOVES:
                        if m2 != INVERSE_MOVES[m1]:
                            s2 = GETTERS[m2](s1)
                            if s2 in self.basin.paths_to_solved:
                                full_moves = move_history + [m1, m2] + self.basin.paths_to_solved[s2]
                                return SolveResult(
                                    solved=True,
                                    moves=full_moves,
                                    n_macro_steps=step,
                                    duration_seconds=time.perf_counter() - t0,
                                    initial_state=initial_state,
                                    final_state=apply_moves(s1, [m2] + self.basin.paths_to_solved[s2]),
                                    history=history,
                                )

                # Depth 3 check (1,176 paths)
                for path in self.paths:
                    if last_move and path[0] == INVERSE_MOVES[last_move]:
                        continue
                    s3 = apply_moves(curr_state, path)
                    if s3 in self.basin.paths_to_solved:
                        full_moves = move_history + list(path) + self.basin.paths_to_solved[s3]
                        return SolveResult(
                            solved=True,
                            moves=full_moves,
                            n_macro_steps=step,
                            duration_seconds=time.perf_counter() - t0,
                            initial_state=initial_state,
                            final_state=apply_moves(curr_state, list(path) + self.basin.paths_to_solved[s3]),
                            history=history,
                        )

                # Outside basin: Evaluate all 3-step candidate routes via neural network
                scores, end_states = self._evaluate_macro_routes(curr_state)

                # Select top candidates, filtering boundary inverses and repeated states
                topk = torch.topk(scores, min(beam_width * 8, len(scores)))
                for score_val, idx_val in zip(topk.values.tolist(), topk.indices.tolist()):
                    path_moves = list(self.paths[idx_val])
                    # Prune boundary inverse (undoing last move)
                    if last_move and path_moves[0] == INVERSE_MOVES[last_move]:
                        continue
                    next_state = end_states[idx_val]
                    # Cycle prevention: do not revisit already visited states in this path
                    if next_state in visited:
                        continue
                    candidates.append((
                        cum_score + score_val,
                        next_state,
                        move_history + path_moves,
                        visited | {next_state},
                    ))

            # Prune beam to beam_width, ensuring unique cube states across the beam
            candidates.sort(key=lambda x: x[0], reverse=True)
            new_beam: List[Tuple[float, Tuple[str, ...], List[str], set]] = []
            seen_states = set()
            for item in candidates:
                st = item[1]
                if st not in seen_states:
                    seen_states.add(st)
                    new_beam.append(item)
                    if len(new_beam) == beam_width:
                        break

            # Fallback if all candidates were pruned
            if not new_beam:
                candidates.sort(key=lambda x: x[0], reverse=True)
                new_beam = candidates[:beam_width]

            beam = new_beam
            best_score, best_state, best_moves, _ = beam[0]
            history.append({
                "step": step,
                "best_score": best_score / step,
                "moves_so_far": len(best_moves),
            })

            if best_state == solved_cube:
                return SolveResult(
                    solved=True,
                    moves=best_moves,
                    n_macro_steps=step,
                    duration_seconds=time.perf_counter() - t0,
                    initial_state=initial_state,
                    final_state=best_state,
                    history=history,
                )

        # If max_macro_steps reached without reaching solved
        best_score, best_state, best_moves, _ = beam[0]
        return SolveResult(
            solved=(best_state == solved_cube),
            moves=best_moves,
            n_macro_steps=max_macro_steps,
            duration_seconds=time.perf_counter() - t0,
            initial_state=initial_state,
            final_state=best_state,
            history=history,
        )

    def _evaluate_macro_routes(
        self, state: Tuple[str, ...]
    ) -> Tuple[torch.Tensor, List[Tuple[str, ...]]]:
        """
        Evaluate all candidate 3-move routes using the 3-step windowed average similarity.
        Uses deduplicated unique states (1,194 instead of 3,528) for 3x faster inference.
        """
        curr_ids = torch.tensor(
            _state_to_ids(state), dtype=torch.long, device=self.device
        )

        # Vectorized state generation for all 1,194 unique candidate states
        all_states = curr_ids[self.all_perms]

        with torch.inference_mode():
            all_embs, _ = self.model(all_states)
            all_embs = F.normalize(all_embs, p=2, dim=-1)

            # Compute similarities according to target_mode
            if self.target_mode == "basin_max":
                all_sims = torch.mm(all_embs, self.basin_embs.T).max(dim=-1).values
            elif self.target_mode == "basin_mean":
                all_sims = torch.mm(all_embs, self.basin_embs.T).mean(dim=-1)
            else:  # 'solved'
                all_sims = (all_embs * self.solved_emb).sum(dim=-1)

            sim1 = all_sims[self.idx1]
            sim2 = all_sims[self.len_u1 + self.idx2]
            sim3 = all_sims[self.len_u1 + self.len_u2 + self.idx3]

            scores = (sim1 + sim2 + sim3) / 3.0

        end_states = [apply_moves(state, p) for p in self.paths]
        return scores, end_states

    def solve_scramble(
        self,
        n_moves: int = 6,
        max_macro_steps: int = 15,
        beam_width: int = 1,
    ) -> Tuple[SolveResult, List[str]]:
        """
        Convenience helper to scramble a solved cube and solve it.

        Args:
            n_moves: Number of random scramble moves.
            max_macro_steps: Maximum macro steps.
            beam_width: Beam width.

        Returns:
            Tuple of (SolveResult, scramble_moves).
        """
        scrambled_cube, scramble_moves = scramble(self.basin.solved, n_moves=n_moves)
        result = self.solve(
            scrambled_cube, max_macro_steps=max_macro_steps, beam_width=beam_width
        )
        return result, scramble_moves
