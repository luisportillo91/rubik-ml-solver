#!/usr/bin/env python3
"""Speed test benchmark for Rubik's cube scramble generation."""

import argparse
import statistics
import time
from typing import Any, Dict, List
import numpy as np

from rubik.cube import scramble


def benchmark_scramble(n_moves: int, repetitions: int = 20, avoid_repeated: bool = False) -> Dict[str, Any]:
    times: List[float] = []
    # Warm-up
    scramble(n_moves=min(n_moves, 10), avoid_repeated=avoid_repeated)

    for _ in range(repetitions):
        start = time.perf_counter()
        scramble(n_moves=n_moves, avoid_repeated=avoid_repeated)
        times.append(time.perf_counter() - start)

    mean_s = statistics.mean(times)
    min_s = min(times)
    moves_per_sec = n_moves / mean_s if mean_s > 0 else float("inf")
    time_per_move_us = (mean_s / n_moves) * 1_000_000

    return {
        "mean_s": mean_s,
        "min_s": min_s,
        "time_per_move_us": time_per_move_us,
        "moves_per_sec": moves_per_sec,
    }


def format_duration(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:7.2f} ms"
    else:
        return f"{seconds:7.3f} s "


def run_benchmarks(move_counts: List[int], repetitions: int, avoid_repeated: bool):
    print("=" * 82)
    print(f" RUBIK CUBE SCRAMBLE SPEED TEST (avoid_repeated={avoid_repeated})")
    print(f" Architecture: Pure Functional Stateless Permutations")
    print("=" * 82)
    header = (
        f"{'Moves':>7} | {'Runs':>5} | {'Mean Total':>12} | "
        f"{'Min Total':>12} | {'Time / Move':>13} | {'Moves / Sec':>13}"
    )
    print(header)
    print("-" * 82)

    for n in move_counts:
        result = benchmark_scramble(n, repetitions=repetitions, avoid_repeated=avoid_repeated)
        mean_str = format_duration(result["mean_s"])
        min_str = format_duration(result["min_s"])
        per_move_str = f"{result['time_per_move_us']:7.2f} µs"
        rate_str = f"{result['moves_per_sec']:11.1f} /s"

        print(
            f"{n:7d} | {repetitions:5d} | "
            f"{mean_str:>12} | {min_str:>12} | {per_move_str:>13} | {rate_str:>13}"
        )

    print("=" * 82)


def main():
    parser = argparse.ArgumentParser(description="Speed test Rubik's cube scramble generation.")
    parser.add_argument(
        "--moves",
        nargs="+",
        type=int,
        default=[10, 100, 1000],
        help="List of move counts to benchmark (default: 10 100 1000)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
        help="Number of repetitions per move count (default: 20)",
    )
    parser.add_argument(
        "--avoid-repeated",
        action="store_true",
        help="Enable avoid_repeated flag during scramble",
    )

    args = parser.parse_args()
    run_benchmarks(args.moves, args.repetitions, args.avoid_repeated)


if __name__ == "__main__":
    main()
