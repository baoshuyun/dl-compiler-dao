# NPU Register Map & Programming Guide

## Overview

The NPU is a memory-mapped coprocessor with a 256-entry × 32-bit instruction
memory. The host CPU controls it via a simplified bus interface.

## Bus Interface

| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| `host_start` | 1 | In | Assert high for 1 cycle to begin execution |
| `npu_done` | 1 | Out | Asserted when program completes |
| `instr_mem[0:255]` | 32 | In | Pre-loaded instruction memory |
| `ext_req` | 1 | Out | DMA request to external memory |
| `ext_grant` | 1 | In | DMA grant from memory controller |
| `ext_rw` | 1 | Out | 0=read, 1=write |
| `ext_addr[9:0]` | 10 | Out | External memory address |
| `ext_wdata[31:0]` | 32 | Out | Write data to external memory |
| `ext_rdata[31:0]` | 32 | In | Read data from external memory |
| `ext_valid` | 1 | In | External memory data valid |
| `debug_results[0:15]` | 32×16 | Out | PE output debug port |

## Instruction Set (32-bit fixed-width)

```
[31:28]  Opcode
[27:0]   Instruction-specific fields
```

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x0 | NOP | — | No operation, advance PC |
| 0x1 | LOAD | `bank[27:26] size[25:16] addr[15:0]` | DMA: ext_mem → SRAM bank |
| 0x2 | STORE | `bank[27:26] size[25:16] addr[15:0]` | DMA: SRAM bank → ext_mem |
| 0x3 | COMPUTE | `a[27:26] b[25:24] c[23:22] dim[21:20] act[19:18]` | Systolic array matmul |
| 0x4 | BARRIER | `type[27:26]` (01=DMA, 10=COMPUTE, 11=ALL) | Pipeline sync |
| 0x5 | CONFIG | (reserved) | Future use |
| 0x6 | LOOP | `count[15:0]` | Repeat next N instructions |

### COMPUTE Field Details

| Field | Bits | Values |
|-------|------|--------|
| A source bank | [27:26] | 0–3 |
| B source bank | [25:24] | 0–3 |
| C destination bank | [23:22] | 0–3 |
| Matrix dimension | [21:20] | 0=4×4, 1=2×2, 2=1×1 |
| Activation | [19:18] | 0=none, 1=ReLU |

## SRAM Map (4 banks × 1024 words = 16 KB)

| Bank | Purpose | Ping/Pong |
|------|---------|-----------|
| 0 | Matrix A (ping) | ✓ |
| 1 | Matrix B (ping) | ✓ |
| 2 | Matrix A (pong) / Result | ✓ |
| 3 | Matrix B (pong) / Scratchpad | ✓ |

## Programming Flow

1. Load input data into external memory (DRAM)
2. Write NPU program into `instr_mem[0:255]`
3. Assert `host_start` (1 cycle pulse)
4. Wait for `npu_done` assertion
5. Read results from external memory at the STORE addresses

### Example: 4×4 MatMul + ReLU

```
External memory layout:
  ext_mem[0x00..0x0F]  ← Matrix A (4×4, row-major)
  ext_mem[0x40..0x4F]  ← Matrix B (4×4, row-major)
  ext_mem[0x80..0x8F]  ← Result C (4×4) — written by NPU

instr_mem:
  [0] LOAD  bank=0, size=16, addr=0x00   // Load A
  [1] LOAD  bank=1, size=16, addr=0x40   // Load B
  [2] BARRIER DMA                          // Wait for loads
  [3] COMPUTE a=0, b=1, c=2, dim=4x4, act=ReLU
  [4] BARRIER COMPUTE                      // Wait for compute
  [5] STORE bank=2, size=16, addr=0x80    // Store result
  [6] BARRIER DMA                          // Wait for store
  [7] NOP                                  // Halt
```

## Pipeline Latency

- **DMA LOAD/STORE:** 1 + size cycles (1 word/cycle after grant)
- **COMPUTE 4×4:** 7 cycles (ROWS + COLS - 1)
- **COMPUTE 2×2:** 3 cycles
- **COMPUTE 1×1:** 1 cycle

## Known Limitations

1. COMPUTE reads from SRAM offset 0 only (no address field)
2. No accumulate mode for K-tiling
3. 32-bit integer only (no FP16/BF16 in current rev)
4. Single-core (no multi-NPU parallelism)
5. Simplified bus (not full AXI)
