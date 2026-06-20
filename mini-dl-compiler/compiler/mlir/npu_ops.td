//===----------------------------------------------------------------------===//
// NPU Operations — TableGen Definition
// 7 opcodes = 7 MLIR operations
//===----------------------------------------------------------------------===//

#ifndef NPU_OPS
#define NPU_OPS

include "npu_dialect.td"

//===----------------------------------------------------------------------===//
// npu.nop
//===----------------------------------------------------------------------===//

def NPU_NopOp : NPU_Op<"nop"> {
  let summary = "No operation, advance PC by 1";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// npu.load
//===----------------------------------------------------------------------===//

def NPU_LoadOp : NPU_Op<"load"> {
  let summary = "DMA transfer from external DRAM to SRAM bank";
  let description = [{
    Copies [base_addr, base_addr + size) from external memory
    into the specified SRAM bank (0-3).
    Supports Ping/Pong double buffering via bank selection.
  }];
  let arguments = (ins
    I32Attr:$bank,        // SRAM bank [0,3]
    I32Attr:$base_addr,   // external memory base address [0,2^16)
    I32Attr:$size         // transfer size in words [1,2^10)
  );
  let assemblyFormat = "$bank `,` $base_addr `,` $size attr-dict";
}

//===----------------------------------------------------------------------===//
// npu.store
//===----------------------------------------------------------------------===//

def NPU_StoreOp : NPU_Op<"store"> {
  let summary = "DMA transfer from SRAM bank to external DRAM";
  let arguments = (ins
    I32Attr:$bank,
    I32Attr:$base_addr,
    I32Attr:$size
  );
  let assemblyFormat = "$bank `,` $base_addr `,` $size attr-dict";
}

//===----------------------------------------------------------------------===//
// npu.compute
//===----------------------------------------------------------------------===//

def NPU_ComputeOp : NPU_Op<"compute"> {
  let summary = "Systolic array matrix multiplication with optional activation";
  let description = [{
    4x4 weight-stationary systolic array.
    A matrix (weights): left-to-right broadcast across rows.
    B matrix (activations): top-to-bottom broadcast down columns.
    C matrix: accumulated in place.
    Pipeline latency = ROWS + COLS - 1 = 7 cycles for 4x4.
    After the pipeline fills, produces 1 result per cycle.
  }];
  let arguments = (ins
    I32Attr:$a_src,       // A matrix source bank [0,3]
    I32Attr:$b_src,       // B matrix source bank [0,3]
    I32Attr:$c_dst,       // C result destination bank [0,3]
    NPU_MatDim:$dim,      // sub-array dimension (4x4/2x2/1x1)
    NPU_ActType:$act      // activation applied after compute
  );
  let assemblyFormat = "$a_src `,` $b_src `,` $c_dst `,` $dim `,` $act attr-dict";
}

//===----------------------------------------------------------------------===//
// npu.barrier
//===----------------------------------------------------------------------===//

def NPU_BarrierOp : NPU_Op<"barrier"> {
  let summary = "Pipeline barrier — stall PC until target engine idle";
  let description = [{
    DMA:     wait until dma_busy == 0
    COMPUTE: wait until cmp_busy == 0
    ALL:     wait until both engines idle
    Prevents RAW and WAW hazards between DMA and COMPUTE engines.
  }];
  let arguments = (ins
    NPU_BarrierType:$barrier_type
  );
  let assemblyFormat = "$barrier_type attr-dict";
}

//===----------------------------------------------------------------------===//
// npu.loop
//===----------------------------------------------------------------------===//

def NPU_LoopOp : NPU_Op<"loop"> {
  let summary = "Hardware loop — repeat next instruction count times";
  let description = [{
    Loads loop_count and repeats the immediately following instruction.
    Single-nested only — no nested loop support in hardware.
    The decoder stores one loop_start_pc and loop_counter register.
  }];
  let arguments = (ins
    I32Attr:$count
  );
  let assemblyFormat = "$count attr-dict";
}

//===----------------------------------------------------------------------===//
// npu.config
//===----------------------------------------------------------------------===//

def NPU_ConfigOp : NPU_Op<"config"> {
  let summary = "Write hardware parameter register (lower 16 bits)";
  let arguments = (ins
    I32Attr:$value
  );
  let assemblyFormat = "$value attr-dict";
}

#endif // NPU_OPS
