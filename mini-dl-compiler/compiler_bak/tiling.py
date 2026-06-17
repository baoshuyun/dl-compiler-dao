# mini-dl-compiler: Tiling Cost Model
# Cache-miss estimation and tile size search for loop tiling optimization
# Operates on matmul loops: M, N, K dimensions -> tile sizes (tm, tn, tk)

from math import ceil
from typing import Tuple


class CostModel:
    """Estimates L1 cache misses for a tiled matrix multiplication.

    For a matmul C[M,N] += A[M,K] * B[K,N] tiled with (tm, tn, tk):
    - Each tile loads tm*K + K*tn + tm*tn elements (A tile + B tile + C tile).
    - If the tile footprint exceeds L1 capacity, a penalty factor is applied.
    - num_tiles = ceil(M/tm) * ceil(N/tn) * ceil(K/tk)
    - total_misses ~ num_tiles * tile_data * penalty

    This is a simplified analytical model; production compilers use
    ML-based cost models (XGBoost, GNN) trained on hardware measurements.
    """

    def __init__(self, l1_capacity_elements: int = 4096):
        """Initialize with L1 cache capacity in elements (default: 4096 floats = 16KB)."""
        self.l1_capacity = l1_capacity_elements

    def estimate_matmul_misses(
        self,
        M: int, N: int, K: int,
        tm: int, tn: int, tk: int,
    ) -> float:
        """Estimate L1 cache misses for given tile configuration.

        Args:
            M, N, K: Problem dimensions.
            tm, tn, tk: Tile sizes for each dimension.

        Returns:
            Estimated total cache misses (lower is better).
        """
        tile_data = tm * tk + tk * tn + tm * tn  # A_tile + B_tile + C_tile footprint
        l1_penalty = 1.0 if tile_data <= self.l1_capacity else 10.0
        num_tiles = ceil(M / tm) * ceil(N / tn) * ceil(K / tk)
        return num_tiles * tile_data * l1_penalty


def compute_tile_sizes(
    op, config: dict = None
) -> Tuple[int, int, int]:
    """Search for near-optimal tiling factors for a matmul operation.

    Exhaustively evaluates candidate tile sizes from [8, 16, 32, 64, 128]
    on each dimension and selects the configuration with minimum estimated
    cache misses.

    Args:
        op: A linalg.matmul Operation.
        config: Optional hardware config dict (l1_capacity, ...).

    Returns:
        Tuple of (tm, tn, tk) tile sizes.
    """
    M, K = op.inputs[0].type.shape
    N = op.inputs[1].type.shape[1]

    model = CostModel(
        l1_capacity_elements=config.get("l1_capacity", 4096) if config else 4096
    )

    candidates = [8, 16, 32, 64, 128]
    best_tiles = (32, 32, 32)
    best_misses = float("inf")

    for tm in candidates:
        for tn in candidates:
            for tk in candidates:
                misses = model.estimate_matmul_misses(M, N, K, tm, tn, tk)
                if misses < best_misses:
                    best_misses = misses
                    best_tiles = (tm, tn, tk)

    return best_tiles
