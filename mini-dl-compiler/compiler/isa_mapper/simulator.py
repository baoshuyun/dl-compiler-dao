"""Python behavioral model of the NPU hardware.

A cycle-accurate simulator that executes NPU instruction programs
and produces numerical results.  Used as the golden reference model
for validating both the ISA Mapper output and the Verilog RTL.

Models:
  - Decoder: sequential instruction fetch & decode
  - DMA Engine: data transfer between external DRAM and SRAM
  - SRAM: 4 banks with ping/pong double buffering
  - Systolic Array: weight-stationary 4x4, 7-cycle pipeline
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .isa import (
    ACT_RELU,
    BAR_DMA,
    BAR_COMPUTE,
    BAR_ALL,
    OP_BARRIER,
    OP_COMPUTE,
    OP_LOAD,
    OP_LOOP,
    OP_NOP,
    OP_STORE,
    PIPELINE_CYCLES,
)
from .types import HardwareConfig, NPUProgram


class NPUSimulator:
    """Cycle-accurate NPU behavioral model.

    Usage::

        sim = NPUSimulator(config)
        sim.load_program(program)
        sim.load_external_memory({addr: data, ...})
        sim.run()
        results = sim.read_external_memory(start_addr, count)
    """

    def __init__(self, config: HardwareConfig | None = None) -> None:
        self.config = config or HardwareConfig()
        self.array_rows = self.config.array_rows
        self.array_cols = self.config.array_cols
        self.num_banks = self.config.sram_banks
        self.bank_depth = self.config.bank_depth
        self.pipeline_cycles = self.config.pipeline_cycles

        # State
        self.instr_mem: list[int] = []
        self.sram: dict[int, NDArray[np.int32]] = {}  # bank_idx → array
        self.ext_mem: dict[int, int] = {}              # addr → value

        self.pc: int = 0
        self.running: bool = False
        self.dma_busy: bool = False
        self.cmp_busy: bool = False
        self.cmp_cycle: int = 0
        self.done: bool = False

        # Loop state
        self.loop_active: bool = False
        self.loop_start_pc: int = 0
        self.loop_counter: int = 0

        # Cycle counter
        self.total_cycles: int = 0

        # Initialize SRAM banks
        for b in range(self.num_banks):
            self.sram[b] = np.zeros(self.bank_depth, dtype=np.int32)

        # Systolic array internal state
        self._pe_acc: NDArray[np.int32] = np.zeros(
            (self.array_rows, self.array_cols), dtype=np.int32
        )
        self._a_pipe: NDArray[np.int32] = np.zeros(
            (self.array_rows, self.array_cols), dtype=np.int32
        )
        self._b_pipe: NDArray[np.int32] = np.zeros(
            (self.array_rows, self.array_cols), dtype=np.int32
        )
        self._pe_valid: NDArray[np.bool_] = np.zeros(
            (self.array_rows, self.array_cols), dtype=np.bool_
        )
        self._cmp_act: int = 0
        self._cmp_mat_dim: int = 0
        self._cmp_a_src: int = 0
        self._cmp_b_src: int = 0
        self._cmp_c_dst: int = 0

    # ── Public API ───────────────────────────────────────────────

    def load_program(self, program: NPUProgram) -> None:
        """Load an instruction program into the simulator."""
        self.instr_mem = list(program.instr_mem)
        self._reset()

    def load_external_memory(self, data: dict[int, int]) -> None:
        """Write values to the simulated external DRAM."""
        self.ext_mem.update(data)

    def read_external_memory(self, start: int, count: int) -> list[int]:
        """Read values from the simulated external DRAM."""
        return [self.ext_mem.get(start + i, 0) for i in range(count)]

    def read_sram(self, bank: int, start: int, count: int) -> NDArray[np.int32]:
        """Read values from an SRAM bank."""
        return self.sram[bank][start:start + count].copy()

    def run(self) -> None:
        """Execute the loaded program until completion."""
        self.running = True
        self.pc = 0
        self.total_cycles = 0

        while self.running:
            self._step()
            self.total_cycles += 1
            # Safety: prevent infinite loops
            if self.total_cycles > 1_000_000:
                raise RuntimeError("NPU simulator exceeded 1M cycles — infinite loop?")

    def run_cycle(self) -> bool:
        """Execute a single cycle. Returns True if still running."""
        self._step()
        self.total_cycles += 1
        return self.running

    # ── Core execution ──────────────────────────────────────────

    def _reset(self) -> None:
        self.pc = 0
        self.running = False
        self.dma_busy = False
        self.cmp_busy = False
        self.cmp_cycle = 0
        self.done = False
        self.loop_active = False
        self.total_cycles = 0

    def _step(self) -> None:
        """Execute one cycle of the NPU."""
        # Update compute pipeline (if active)
        if self.cmp_busy:
            self._compute_step()

        if self.pc >= len(self.instr_mem):
            self.running = False
            return

        instr = self.instr_mem[self.pc]
        opcode = (instr >> 28) & 0xF

        if opcode == OP_NOP:
            self.pc += 1

        elif opcode == OP_LOAD:
            if not self.dma_busy:
                self._execute_load(instr)
                self.pc += 1

        elif opcode == OP_STORE:
            if not self.dma_busy:
                self._execute_store(instr)
                self.pc += 1

        elif opcode == OP_COMPUTE:
            if not self.cmp_busy:
                self._start_compute(instr)
                self.pc += 1

        elif opcode == OP_BARRIER:
            self._execute_barrier(instr)
            # PC advances only if barrier passes
            if self._barrier_satisfied(instr):
                self.pc += 1

        elif opcode == OP_LOOP:
            self._execute_loop(instr)
            self.pc += 1

        else:
            # Unknown opcode — treat as NOP
            self.pc += 1

        # Auto-halt check
        if self.pc >= len(self.instr_mem) - 1:
            self.running = False
            self.done = True

    # ── Instruction implementations ─────────────────────────────

    def _execute_load(self, instr: int) -> None:
        bank = (instr >> 26) & 0x3
        size = (instr >> 16) & 0x3FF
        ext_addr = instr & 0xFFFF

        for i in range(size):
            self.sram[bank][i] = np.int32(self.ext_mem.get(ext_addr + i, 0))
        self.dma_busy = False

    def _execute_store(self, instr: int) -> None:
        bank = (instr >> 26) & 0x3
        size = (instr >> 16) & 0x3FF
        ext_addr = instr & 0xFFFF

        for i in range(size):
            self.ext_mem[ext_addr + i] = int(self.sram[bank][i])

    def _start_compute(self, instr: int) -> None:
        self._cmp_a_src = (instr >> 26) & 0x3
        self._cmp_b_src = (instr >> 24) & 0x3
        self._cmp_c_dst = (instr >> 22) & 0x3
        mat_dim = (instr >> 20) & 0x3
        self._cmp_act = (instr >> 18) & 0x3
        self._cmp_mat_dim = mat_dim

        self.cmp_busy = True
        self.cmp_cycle = 0
        self._pe_acc.fill(0)

    def _compute_step(self) -> None:
        """Advance the systolic array pipeline by one cycle.

        Functional model: counts cycles up to pipeline_cycles, then
        performs the matrix multiplication as a direct computation.
        This is a golden reference model intended to match the
        functional result of the Verilog RTL, not its exact internal
        data flow.
        """
        self.cmp_cycle += 1

        if self.cmp_cycle >= self.pipeline_cycles:
            self._compute_result()
            self.cmp_busy = False
            self.cmp_cycle = 0

    def _compute_result(self) -> None:
        """Execute the matrix multiply functionally.

        Reads A and B from their source SRAM banks, computes
        C = A @ B (row-major layout), applies activation, and writes
        the result to the destination SRAM bank.
        """
        if self._cmp_mat_dim == 0:
            rows, cols, k_dim = 4, 4, 4
        elif self._cmp_mat_dim == 1:
            rows, cols, k_dim = 2, 2, 2
        else:
            rows, cols, k_dim = 1, 1, 1

        a_bank = self.sram[self._cmp_a_src]
        b_bank = self.sram[self._cmp_b_src]
        c_bank = self.sram[self._cmp_c_dst]

        # Extract matrices from SRAM row-major layout
        # A: rows × k_dim, stored row-major
        A = np.zeros((rows, k_dim), dtype=np.int32)
        for r in range(rows):
            for k in range(k_dim):
                idx = r * k_dim + k
                if idx < self.bank_depth:
                    A[r, k] = a_bank[idx]

        # B: k_dim × cols, stored row-major
        B = np.zeros((k_dim, cols), dtype=np.int32)
        for k in range(k_dim):
            for c in range(cols):
                idx = k * cols + c
                if idx < self.bank_depth:
                    B[k, c] = b_bank[idx]

        # C = A @ B
        C = A.astype(np.int64) @ B.astype(np.int64)

        # Activation
        if self._cmp_act == ACT_RELU:
            C = np.maximum(C, 0)

        # Write back row-major
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx < self.bank_depth:
                    c_bank[idx] = np.int32(C[r, c])

    def _execute_barrier(self, instr: int) -> None:
        """Barrier check: PC stalls until condition satisfied."""
        bar_type = (instr >> 26) & 0x3
        # Stall is implicit: _step() won't advance PC if _barrier_satisfied is False

    def _barrier_satisfied(self, instr: int) -> bool:
        bar_type = (instr >> 26) & 0x3
        if bar_type == BAR_DMA:
            return not self.dma_busy
        if bar_type == BAR_COMPUTE:
            return not self.cmp_busy
        if bar_type == BAR_ALL:
            return not self.dma_busy and not self.cmp_busy
        return True

    def _execute_loop(self, instr: int) -> None:
        count = instr & 0xFFFF
        if not self.loop_active:
            self.loop_active = True
            self.loop_start_pc = self.pc + 1
            self.loop_counter = count
        else:
            self.loop_counter -= 1
            if self.loop_counter > 0:
                self.pc = self.loop_start_pc - 1  # -1 because _step will +1
            else:
                self.loop_active = False
