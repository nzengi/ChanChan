#!/usr/bin/env python3
"""
Benchmark script for Chan-ZKP.

Measures success rate and runtime across different (n, p, iterations) settings.
Uses real protocol flow (no mocks) with blinding enabled by default.
"""

import sys
import os
import time
from typing import List, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import MathEngine, ColorOracle
from src.actors import Prover, Verifier, Challenge, Response


def run_single(n: int, p: int, iterations: int, max_attempts: int = 5000):
    engine = MathEngine(dimension=n, modulus=p)
    prover = Prover(engine, verbose=False)
    verifier = Verifier(engine, verbose=False)

    success = 0
    total = iterations

    t0 = time.perf_counter()

    for _ in range(iterations):
        challenge = verifier.generate_challenge()
        try:
            v_valid, _ = prover.find_valid_green_for_challenge(
                challenge, max_attempts=max_attempts
            )
            prover.secret_v = v_valid

            commitment = prover.commit()
            response = prover.solve_challenge(challenge)

            result = verifier.verify(commitment, response, reveal_v=True)
            if result.is_valid:
                success += 1
        except Exception:
            # Treat as failure for benchmark purposes
            pass

    elapsed = time.perf_counter() - t0
    return {
        "n": n,
        "p": p,
        "iterations": iterations,
        "success": success,
        "total": total,
        "success_rate": success / total * 100.0,
        "elapsed_sec": elapsed,
        "per_iter_ms": (elapsed / total) * 1000.0,
    }


def pretty_print(results: List[dict]):
    print("\nBenchmark Results")
    print("-" * 70)
    print(
        f"{'n':>3} {'p':>5} {'iters':>6} {'succ':>6} {'rate%':>7} {'time(s)':>8} {'per_iter(ms)':>13}"
    )
    for r in results:
        print(
            f"{r['n']:>3} {r['p']:>5} {r['iterations']:>6} {r['success']:>6} "
            f"{r['success_rate']:>6.1f} {r['elapsed_sec']:>8.3f} {r['per_iter_ms']:>13.3f}"
        )
    print("-" * 70)


def main():
    configs: List[Tuple[int, int, int]] = [
        (4, 7, 50),      # demo small
        (5, 11, 50),     # medium
        (6, 13, 30),     # larger
        (8, 17, 20),     # big demo
        (10, 23, 20),    # bigger field
        (4, 2305843009213693951, 5),  # ~2^61 prime, large p demo
    ]

    results = []
    for n, p, iters in configs:
        res = run_single(n, p, iters, max_attempts=5000)
        results.append(res)

    pretty_print(results)


if __name__ == "__main__":
    main()

