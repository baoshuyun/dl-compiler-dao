//===----------------------------------------------------------------------===//
// NPU Dialect Definition — TableGen
// 7-instruction NPU ISA mapped to MLIR dialect
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT
#define NPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// NPU Dialect
//===----------------------------------------------------------------------===//

def NPU_Dialect : Dialect {
  let name = "npu";
  let summary = "NPU instruction set dialect";
  let description = [{
    Maps NPU ISA to MLIR operations:
      npu.load    — DMA: DRAM -> SRAM bank
      npu.store   — DMA: SRAM bank -> DRAM
      npu.compute — Systolic array matmul
      npu.barrier — Pipeline synchronization
      npu.config  — Hardware parameter register write
      npu.loop    — Hardware-level loop (single-nested only)
      npu.nop     — No operation, advance PC
  }];
  let cppNamespace = "::mlir::npu";
}

//===----------------------------------------------------------------------===//
// NPU Enums
//===----------------------------------------------------------------------===//

def NPU_ActNone  : I32EnumAttrCase<"NONE",  0, "none">;
def NPU_ActRelu  : I32EnumAttrCase<"RELU",  1, "relu">;
def NPU_ActType  : I32EnumAttr<"Activation", "Post-compute activation", [
  NPU_ActNone, NPU_ActRelu
]>;

def NPU_Dim4x4 : I32EnumAttrCase<"MAT_4x4", 0, "4x4">;
def NPU_Dim2x2 : I32EnumAttrCase<"MAT_2x2", 1, "2x2">;
def NPU_Dim1x1 : I32EnumAttrCase<"MAT_1x1", 2, "1x1">;
def NPU_MatDim : I32EnumAttr<"MatDim", "Matrix dimension for sub-array", [
  NPU_Dim4x4, NPU_Dim2x2, NPU_Dim1x1
]>;

def NPU_BarDMA     : I32EnumAttrCase<"DMA",     1, "dma">;
def NPU_BarCompute : I32EnumAttrCase<"COMPUTE", 2, "compute">;
def NPU_BarAll     : I32EnumAttrCase<"ALL",     3, "all">;
def NPU_BarrierType : I32EnumAttr<"BarrierType", "Synchronization target", [
  NPU_BarDMA, NPU_BarCompute, NPU_BarAll
]>;

#endif // NPU_DIALECT
