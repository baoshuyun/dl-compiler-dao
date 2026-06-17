"""Assembler: NPUInstruction list → NPUProgram (binary instr_mem).

Packs decoded instructions into a 256-element 32-bit word array,
ready to load into the NPU hardware or simulator.
"""

from __future__ import annotations

from .isa import INSTR_DEPTH, OP_NOP, encode_nop
from .types import NPUInstruction, NPUProgram


class Assembler:
    """Assembles NPUInstruction list into a binary NPUProgram.

    Usage::

        asm = Assembler()
        program = asm.assemble(instructions)
        print(program.display())
    """

    MAX_INSTRUCTIONS = INSTR_DEPTH  # 256

    def assemble(self, instructions: list[NPUInstruction]) -> NPUProgram:
        """Pack instructions into a 256-word instruction memory.

        Args:
            instructions: Ordered list of decoded instructions.

        Returns:
            NPUProgram with binary instr_mem (length 256).

        Raises:
            ValueError: If more than 256 instructions are provided.
        """
        if len(instructions) > self.MAX_INSTRUCTIONS:
            raise ValueError(
                f"Too many instructions: {len(instructions)} > {self.MAX_INSTRUCTIONS}"
            )

        # Build instruction memory
        instr_mem = list(self._pack_instructions(instructions))

        # Pad to 256 with NOPs
        nop = encode_nop()
        while len(instr_mem) < self.MAX_INSTRUCTIONS:
            instr_mem.append(nop)

        return NPUProgram(
            instructions=tuple(instructions),
            instr_mem=tuple(instr_mem),
        )

    @staticmethod
    def _pack_instructions(
        instructions: list[NPUInstruction],
    ) -> list[int]:
        """Extract binary words from decoded instructions."""
        return [instr.binary for instr in instructions]

    @staticmethod
    def disassemble(program: NPUProgram) -> str:
        """Return a human-readable disassembly of an NPUProgram."""
        lines = []
        for i, word in enumerate(program.instr_mem):
            opcode = (word >> 28) & 0xF
            if opcode == int(OP_NOP) and i >= program.program_length:
                continue  # Skip trailing NOPs
            op_names = {0: "NOP", 1: "LOAD", 2: "STORE", 3: "COMPUTE",
                        4: "BARRIER", 5: "CONFIG", 6: "LOOP"}
            name = op_names.get(opcode, f"UNK({opcode})")
            lines.append(f"[{i:3d}] {name:<8} 0x{word:08X}")
        return "\n".join(lines)
