"""Entry point for the Mini DL Compiler.

Usage:
    python main.py              # run the demo model
    python main.py --benchmark  # run with benchmarking
    python main.py --no-art     # skip graph visualization
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from compiler.codegen import NumpyCompiler
from compiler.ir import Graph
from compiler.viz import (
    dump_text_graph,
    export_graph_dot,
    export_reachable_dot,
    render_dot,
)

logger = logging.getLogger("mini-dl-compiler")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_demo_model() -> tuple[Graph, np.ndarray[Any, Any]]:
    """Build the demo computation graph and return (graph, test_input)."""
    g = Graph()

    x = g.Input("x")

    W = g.Const([[1, 2, 3], [4, 5, 6]], name="W")
    b = g.Const([1, 0, -1], name="b")

    # These two Const nodes will be folded: 2+3=5
    c = g.Add(
        g.Const([2, 2, 2], name="c1"),
        g.Const([3, 3, 3], name="c2"),
        name="bias_sum",
    )

    # Dead-code branch — not connected to the output
    dead_w = g.Const([[9, 9], [9, 9]], name="dead_w")
    dead_x = g.Const([1, 2], name="dead_x")
    dead_mm = g.MatMul(dead_x, dead_w, name="dead_matmul")
    dead_relu = g.ReLU(dead_mm, name="dead_relu")

    # Main graph: matmul -> add -> relu
    mm = g.MatMul(x, W, name="mm")
    y = g.ReLU(g.Add(mm, g.Add(b, c), name="bias_add"), name="relu_out")

    g.output = y
    return g, np.array([1.0, 2.0])


# ---------------------------------------------------------------------------
# Reference eager implementation
# ---------------------------------------------------------------------------

def eager(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Reference NumPy implementation of the demo model."""
    W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    b = np.array([1, 0, -1], dtype=np.float64)
    c = np.array([5, 5, 5], dtype=np.float64)
    return np.maximum((x @ W) + (b + c), 0)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark(fn: Callable[..., Any], x: np.ndarray[Any, Any], runs: int = 10_000) -> tuple[float, np.ndarray[Any, Any]]:
    """Run *fn(x)* for *runs* iterations.  Returns (elapsed_seconds, last_output)."""
    # Warmup
    out = fn(x)
    start = time.perf_counter()
    for _ in range(runs):
        out = fn(x)
    elapsed = time.perf_counter() - start
    return elapsed, out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini DL Compiler — compile a toy model to NumPy",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="run a simple benchmark comparing compiled vs. eager",
    )
    parser.add_argument(
        "--no-art", action="store_true",
        help="skip DOT/PNG graph export",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    # Build and compile
    graph, x_input = build_demo_model()
    compiler = NumpyCompiler()

    fn, source, optimized, logs = compiler.compile(graph.output, graph=graph)  # type: ignore[arg-type]

    # Optimization logs
    logger.info("Optimization log:")
    for line in logs:
        logger.info("  %s", line)

    # Generated source
    logger.info("Generated code:\n%s", source)

    # Text graph dump
    logger.info(dump_text_graph(optimized))

    # Visualization
    if not args.no_art:
        artifacts = Path("artifacts")
        artifacts.mkdir(exist_ok=True)

        before_dot = export_graph_dot(
            graph, graph.output,  # type: ignore[arg-type]
            path=str(artifacts / "graph_before.dot"),
            title="Original Graph",
        )
        after_dot = export_reachable_dot(
            optimized,
            path=str(artifacts / "graph_after.dot"),
            title="Optimized Graph",
        )

        before_png, ok1, msg1 = render_dot(before_dot, str(artifacts / "graph_before.png"))
        after_png, ok2, msg2 = render_dot(after_dot, str(artifacts / "graph_after.png"))

        logger.info("DOT / PNG files:")
        logger.info("  before: %s", before_dot)
        logger.info("  after:  %s", after_dot)
        logger.info("  before PNG: %s %s", before_png, "(ok)" if ok1 else f"(skip: {msg1})")
        logger.info("  after PNG:  %s %s", after_png, "(ok)" if ok2 else f"(skip: {msg2})")

    # Correctness check
    compiled_out = fn(x_input)
    eager_out = eager(x_input)
    logger.info("Correctness check:")
    logger.info("  input:           %s", x_input)
    logger.info("  compiled output: %s", compiled_out)
    logger.info("  eager output:    %s", eager_out)
    logger.info("  match:           %s", np.allclose(compiled_out, eager_out))

    # Benchmark
    if args.benchmark:
        logger.info("Benchmark (%d runs):", 10_000)
        t_compiled, _ = benchmark(fn, x_input)
        t_eager, _ = benchmark(eager, x_input)
        logger.info("  compiled: %.6f s", t_compiled)
        logger.info("  eager:    %.6f s", t_eager)
        if t_eager > 0:
            speedup = t_eager / t_compiled
            logger.info("  speedup:  %.2fx", speedup)


if __name__ == "__main__":
    main()
