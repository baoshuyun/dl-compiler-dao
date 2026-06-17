"""Tensor tiling: decompose large matrix ops into NPU-sized tiles.

The NPU systolic array is 4×4 (configurable via HardwareConfig).
Any matmul larger than 4×4 must be split across multiple COMPUTE
instructions, with partial sums accumulated in SRAM.
"""

from __future__ import annotations

from .types import ActType, HardwareConfig, MatDim, TileOp


class Tiler:
    """Decompose matrix operations into tiles that fit the systolic array.

    Usage::

        tiler = Tiler(HardwareConfig(array_rows=4, array_cols=4))
        tiles = tiler.tile_matmul("A", "B", "C", M=256, N=256, K=256)
    """

    def __init__(self, config: HardwareConfig | None = None) -> None:
        self.config = config or HardwareConfig()

    # ── Public API ───────────────────────────────────────────────

    def tile_matmul(
        self,
        a_name: str,
        b_name: str,
        c_name: str,
        M: int,
        N: int,
        K: int,
        act: ActType = ActType.NONE,
    ) -> list[TileOp]:
        """Tile a matmul C[M,N] = A[M,K] @ B[K,N].

        Args:
            a_name: Logical buffer name for matrix A.
            b_name: Logical buffer name for matrix B.
            c_name: Logical buffer name for result C.
            M: Rows of A / rows of C.
            N: Columns of B / columns of C.
            K: Columns of A / rows of B (reduction dimension).
            act: Optional activation function applied after matmul.

        Returns:
            Ordered list of TileOp, one per COMPUTE instruction.
            When K > 4, multiple tiles write to the same C region
            (partial sum accumulation).
        """
        tm = self.config.array_rows
        tn = self.config.array_cols
        tk = min(self.config.array_rows, K)  # K-tile matches row dimension

        tiles: list[TileOp] = []

        for m in range(0, M, tm):
            m_end = min(m + tm, M)
            m_size = m_end - m
            for n in range(0, N, tn):
                n_end = min(n + tn, N)
                n_size = n_end - n
                for k in range(0, K, tk):
                    k_end = min(k + tk, K)
                    tile_act = act if k == K - tk or k + tk >= K else ActType.NONE

                    tiles.append(TileOp(
                        op_type="matmul",
                        a_name=a_name,
                        b_name=b_name,
                        c_name=c_name,
                        a_slice=(slice(m, m_end), slice(k, k_end)),
                        b_slice=(slice(k, k_end), slice(n, n_end)),
                        c_slice=(slice(m, m_end), slice(n, n_end)),
                        mat_dim=self._pick_dim(m_size, n_size),
                        act=tile_act,
                        tile_idx=(m // tm, n // tn, k // tk),
                    ))

        return tiles

    def tile_fused_mma_bias_relu(
        self,
        x_name: str,
        w_name: str,
        b_name: str,
        c_name: str,
        M: int,
        N: int,
        K: int,
    ) -> list[TileOp]:
        """Tile a fused matmul+bias+relu C[M,N] = ReLU(X[M,K] @ W[K,N] + b[N]).

        The bias is applied after the final K-tile accumulation, before ReLU.
        """
        tiles = self.tile_matmul(x_name, w_name, c_name, M, N, K, act=ActType.RELU)
        return tiles

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _pick_dim(m_size: int, n_size: int) -> MatDim:
        """Select the smallest MatDim that fits the tile size."""
        if m_size <= 1 and n_size <= 1:
            return MatDim.M1x1
        if m_size <= 2 and n_size <= 2:
            return MatDim.M2x2
        return MatDim.M4x4
