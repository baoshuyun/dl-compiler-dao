"""
MLIR NPU Backend — Dialect definition and lowering pipeline.

Modules:
  npu_dialect         — NPU dialect: 7 opcodes as MLIR operations
  lowering_pipeline   — Full pass pipeline: linalg → tiles → npu → binary
  frontend            — Soft_Stack TaskGraph → linalg dialect
  onnx_frontend       — ONNX model → linalg dialect (PyTorch→ONNX→linalg)
  rvv_lowering        — linalg.matmul → RISC-V Vector (RVV) assembly
  npu_dialect.td / npu_ops.td — TableGen definitions (for C++ MLIR builds)
"""
from .npu_dialect import (
    Opcode, ActType, MatDim, BarrierType,
    NpuOp, NpuNop, NpuLoad, NpuStore,
    NpuCompute, NpuBarrier, NpuConfig, NpuLoop,
    NpuProgram,
)
from .lowering_pipeline import (
    HardwareConfig,
    LinalgMatmul, AffineCopy,
    TileOp,
    LinalgToTilesPass,
    TileToNpuOpsPass,
    NpuOpsToProgramPass,
    NPULoweringPipeline,
)
from .frontend import (
    LinalgProgram,
    TaskGraphToLinalg,
)
from .onnx_frontend import (
    OnnxTensor, OnnxProgram, OnnxToLinalg,
    LinalgConv2D,
)
from .rvv_lowering import (
    RVVConfig, RVVOp, RVVProgram,
    LinalgToRVVPass, RVVLoweringPipeline,
)

__all__ = [
    # Dialect
    "Opcode", "ActType", "MatDim", "BarrierType",
    "NpuOp", "NpuNop", "NpuLoad", "NpuStore",
    "NpuCompute", "NpuBarrier", "NpuConfig", "NpuLoop",
    "NpuProgram",
    # Lowering
    "HardwareConfig",
    "LinalgMatmul", "AffineCopy",
    "TileOp",
    "LinalgToTilesPass",
    "TileToNpuOpsPass",
    "NpuOpsToProgramPass",
    "NPULoweringPipeline",
    # Frontend
    "LinalgProgram",
    "TaskGraphToLinalg",
    # ONNX Frontend
    "OnnxTensor", "OnnxProgram", "OnnxToLinalg",
    "LinalgConv2D",
    # RVV Backend
    "RVVConfig", "RVVOp", "RVVProgram",
    "LinalgToRVVPass", "RVVLoweringPipeline",
]
