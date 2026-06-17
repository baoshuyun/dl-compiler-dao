// isa_defines.vh -- NPU 32-bit ISA definitions
// 7 instruction types + 1 reserved (7+1 total)
// bit[31:28] = opcode, bits[27:0] = instruction-specific fields
// Part of AI_Compiler_Project / NPU_Project

`ifndef ISA_DEFINES_VH
`define ISA_DEFINES_VH

// -- Opcodes (4-bit, bits [31:28]) --

`define OP_NOP      4'h0   // No operation: stall 1 cycle
`define OP_LOAD     4'h1   // DMA load: external DRAM -> SRAM
`define OP_STORE    4'h2   // DMA store: SRAM -> external DRAM
`define OP_COMPUTE  4'h3   // Systolic array computation
`define OP_BARRIER  4'h4   // Synchronization barrier
`define OP_CONFIG   4'h5   // Configure hardware parameters
`define OP_LOOP     4'h6   // Loop: repeat next N instructions

// -- COMPUTE instruction field definitions --
// bit[27:26] = a_src: source bank for matrix A
// bit[25:24] = b_src: source bank for matrix B
// bit[23:22] = c_dst: destination bank for matrix C
// bit[21:20] = mat_dim: matrix dimension {4x4, 2x2, 1x1}
// bit[19:18] = act: activation function {NONE, RELU}

`define SRC_PING    2'b00
`define SRC_PONG    2'b01
`define SRC_BANK0   2'b10
`define SRC_BANK1   2'b11

`define DST_PING    2'b00
`define DST_PONG    2'b01
`define DST_BANK2   2'b10
`define DST_BANK3   2'b11

`define MAT_4x4     2'b00   // 4x4 systolic array (16 PEs)
`define MAT_2x2     2'b01   // 2x2 sub-array (4 PEs)
`define MAT_1x1     2'b10   // 1x1 (single PE, element-wise)

`define ACT_NONE    2'b00   // No activation
`define ACT_RELU    2'b01   // ReLU activation

// -- BARRIER instruction field definitions --
// bit[1:0] = bar_type

`define BAR_DMA      2'b01   // Wait for DMA engine to finish
`define BAR_COMPUTE  2'b10   // Wait for compute array to finish
`define BAR_ALL      2'b11   // Wait for both DMA and compute

// -- CONFIG instruction field definitions --
// bit[15:0] = config_value

// -- LOOP instruction field definitions --
// bit[15:0] = loop_count: number of iterations

`endif // ISA_DEFINES_VH
