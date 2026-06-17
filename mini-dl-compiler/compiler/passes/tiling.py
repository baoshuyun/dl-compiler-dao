"""Tiling pass with analytical cost model + beam search.

Decomposes matmul and fused_mma_bias_relu operations into tiles
that fit within a configurable SRAM budget.  Uses a beam search
over the tile-size space to minimise estimated cache / SRAM misses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ir import Graph, Node
from . import Pass


@dataclass
class TileConfig:
    """A tiling configuration for a matmul."""
    tm: int  # M tile size (rows per tile)
    tn: int  # N tile size (cols per tile)
    tk: int  # K tile size (reduction per tile)
    estimated_misses: int = 0


class CostModel:
    """Analytical cost model for matmul tiling.

    Estimates cache/SRAM misses based on tile data footprint.
    A tile that exceeds the SRAM budget incurs a 10× penalty
    (spill to DRAM).
    """

    def __init__(self, capacity_words: int = 1024) -> None:
        self.capacity = capacity_words

    def estimate(self, M: int, N: int, K: int,
                 tm: int, tn: int, tk: int) -> int:
        """Estimate total SRAM misses for a tiled matmul.

        Args:
            M, N, K: Full matrix dimensions.
            tm, tn, tk: Tile sizes.

        Returns:
            Estimated total misses (lower is better).
        """
        # Number of tiles in each dimension
        mtiles = (M + tm - 1) // tm
        ntiles = (N + tn - 1) // tn
        ktiles = (K + tk - 1) // tk

        # Data footprint of one tile: A[tm,tk] + B[tk,tn] + C[tm,tn]
        tile_footprint = tm * tk + tk * tn + tm * tn

        # Penalty if tile doesn't fit in SRAM
        penalty = 10 if tile_footprint > self.capacity else 1

        # Total: tiles × footprint × penalty
        total_tiles = mtiles * ntiles * ktiles
        return total_tiles * tile_footprint * penalty


class TilingPass(Pass):
    """Decompose matmul nodes into tiled operations.

    Searches candidate tile sizes using beam search, selecting
    the configuration with the minimum estimated SRAM misses.

    Attributes:
        target_rows: Systolic array rows (default 4).
        target_cols: Systolic array columns (default 4).
        sram_capacity: SRAM capacity in 32-bit words (default 1024).
        beam_width: Number of candidates to keep per search step.
    """

    name = "Tiling"

    # Candidate tile sizes — powers of 2 from 2 to 128
    CANDIDATES = (2, 4, 8, 16, 32, 64, 128)

    def __init__(
        self,
        target_rows: int = 4,
        target_cols: int = 4,
        sram_capacity: int = 1024,
        beam_width: int = 3,
    ) -> None:
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.cost_model = CostModel(sram_capacity)
        self.beam_width = beam_width

    def run(self, ir: Graph, **kwargs: Any) -> Graph:
        """Apply tiling to all matmul nodes in the graph.

        Each matmul/fused_mma_bias_relu node gets a 'tile' attribute
        with the optimal (tm, tn, tk) configuration.
        """
        for node in ir.nodes:
            if node.op in ("matmul", "fused_mma_bias_relu"):
                shape = self._infer_shape(node)
                if shape is None:
                    continue
                M, N, K = shape
                best = self._search(M, N, K)
                node.attrs["tile"] = (best.tm, best.tn, best.tk)
                node.attrs["estimated_misses"] = best.estimated_misses

        return ir

    def verify(self, before: Graph, after: Graph) -> list[str]:
        warnings = []
        for node in after.nodes:
            if node.op in ("matmul", "fused_mma_bias_relu"):
                if "tile" not in node.attrs:
                    warnings.append(
                        f"{node.op} node {node.name or 'unnamed'} "
                        f"missing tile config after TilingPass"
                    )
        return warnings

    # ── Internal ─────────────────────────────────────────────────

    def _search(self, M: int, N: int, K: int) -> TileConfig:
        """Beam search over candidate tile sizes."""
        candidates: list[TileConfig] = []

        for tm in self.CANDIDATES:
            if tm > M * 2:  # Don't tile much larger than the dimension
                continue
            for tn in self.CANDIDATES:
                if tn > N * 2:
                    continue
                for tk in self.CANDIDATES:
                    if tk > K * 2:
                        continue
                    misses = self.cost_model.estimate(M, N, K, tm, tn, tk)
                    candidates.append(TileConfig(tm, tn, tk, misses))

        if not candidates:
            return TileConfig(
                min(M, self.target_rows),
                min(N, self.target_cols),
                min(K, self.target_rows),
            )

        # Sort by misses, take top beam_width
        candidates.sort(key=lambda c: c.estimated_misses)
        return candidates[0]

    @staticmethod
    def _infer_shape(node: Node) -> tuple[int, int, int] | None:
        """Infer (M, N, K) from node inputs.

        Returns None if shapes cannot be determined.
        """
        import numpy as np

        shapes: list[tuple[int, ...]] = []
        for inp in node.inputs:
            if isinstance(inp.value, np.ndarray):
                shapes.append(inp.value.shape)
            elif "shape" in inp.attrs:
                shapes.append(inp.attrs["shape"])
            elif inp.op == "input":
                # Input node has no known shape — skip
                shapes.append(())

        if node.op == "fused_mma_bias_relu" and len(shapes) >= 2:
            # x[M,K], w[K,N], b[N] — w is typically Const with known shape
            # Find the weight (Const)
            for i, inp in enumerate(node.inputs):
                if inp.op == "const" and isinstance(inp.value, np.ndarray):
                    w_shape = inp.value.shape
                    if len(w_shape) == 2:
                        K, N = w_shape
                        # M is unknown for input, default to 4
                        M = 4
                        # Try to get M from x if it has shape info
                        for j, other in enumerate(node.inputs):
                            if j != i and isinstance(other.value, np.ndarray) and len(other.value.shape) == 2:
                                M = other.value.shape[0]
                        return (M, N, K)
            return None

        if node.op == "matmul" and len(shapes) == 2:
            s0, s1 = shapes[0], shapes[1]
            # If one operand has known shape, use it
            if len(s0) == 2 and len(s1) == 2:
                return (s0[0], s1[1], s0[1])
            if len(s0) == 2:
                # Only a has shape; assume b is Kx? with K=s0[1]
                return (s0[0], s0[1], s0[1])  # N=K as fallback
            if len(s1) == 2:
                return (s1[1], s1[1], s1[0])  # M=K as fallback

        return None
