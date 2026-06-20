# ADR 001: Linalg-based lowering pipeline for NPU backend

## Status
Accepted (2026-06-20)

## Context
The NPU backend needs a lowering pipeline from high-level ops to hardware instructions. We evaluated two approaches:

**Option A: Direct lowering (Graph IR → NPU instructions)**
- Skip intermediate dialects, emit NPU instructions directly from graph nodes.
- Pros: simple, fewer passes, faster to implement.
- Cons: no reusable intermediate representation; each new op requires full rewrite of lowering logic; hard to add new backends (RVV, LLVM).

**Option B: Linalg dialect lowering (Graph IR → linalg → tiles → NPU)**
- Insert a linalg dialect layer based on MLIR conventions.
- Pros: reusable intermediate dialect; new backends (RVV, LLVM) share the linalg layer; matches MLIR community standards; easier to upstream contributions.
- Cons: more passes, larger codebase, steeper learning curve.

## Decision
Chose **Option B**: lower through a linalg dialect layer.

Rationale:
1. The linalg dialect is the standard entry point for MLIR-based codegen.
2. A separate linalg layer enables parallel backend development (NPU + RVV).
3. The TileOp abstraction decouples hardware geometry from the compute description.
4. TableGen definitions (`npu_dialect.td`) provide a single source of truth for the ISA encoding, keeping Python and Verilog in sync.

## Consequences
- Added `compiler/mlir/` with dialect definitions, lowering pipeline, frontend bridge.
- ONNX frontend lowers directly to linalg — any backend (NPU, RVV, future LLVM) benefits automatically.
- K-tiling for matmul added 30% more code but enables tiling for matrices >4×4.
- RVV backend implemented in 2 hours by reusing the linalg layer.
