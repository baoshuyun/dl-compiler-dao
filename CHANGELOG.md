# Changelog

## [0.3.0] — 2026-06-20

### Added
- `compiler/mlir/onnx_frontend.py` — ONNX model → linalg dialect lowering (MatMul, Gemm, Conv, ReLU, Softmax, Reshape, BatchNorm)
- `compiler/mlir/rvv_lowering.py` — linalg.matmul → RISC-V Vector (RVV v1.0) assembly codegen
- `compiler/mlir/npu_dialect.py` — NPU dialect: 7 opcodes (nop/load/store/compute/barrier/config/loop)
- `compiler/mlir/lowering_pipeline.py` — Full pass pipeline: linalg → tiles → npu → binary
- `compiler/mlir/frontend.py` — Soft_Stack TaskGraph → linalg bridge
- `NPU_Soft_Hard_Stack/npu_stack/backends/npu_mlir.py` — MLIR-based NPU backend
- CI workflow with cross-project test suite (97 + 23 = 120 tests)

### Changed
- MLIR lowering pipeline restored and integrated into active compiler tree

## [0.2.0] — 2026-06-14

### Added
- ISA Mapper: tiler, scheduler, bank allocator, assembler, simulator
- NPU simulator with 4x4 systolic array model
- E2E integration tests across three projects
- Graph → SSA lowering pass
- Memory planning pass with liveness-aware buffer reuse

## [0.1.0] — 2026-05-29

### Added
- Graph IR with 12 node types (input, const, add, matmul, relu, conv2d, pool, batchnorm, softmax, reshape, transpose, fused MMA)
- Optimizer pipeline: constant folding, operator fusion, dead code elimination
- NumPy codegen backend
- Soft_Stack compiler frontend with AST → TaskGraph lowering
- NPU_Project Verilog RTL: decoder, DMA engine, multi-bank SRAM, PE, systolic array
