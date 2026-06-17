"""NPU backend — compiles IR graphs to NPU ISA programs.

Uses the ISA Mapper to translate matmul + activation patterns into
NPU 32-bit instructions, then runs them on the Python NPU simulator
for verification.

Supported ops: matmul, add, relu, fused_mma_bias_relu
Unsupported ops: conv2d, pool, softmax (fall back to numpy)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..ir import Graph, Node
from ..isa_mapper import ISAMapper, NPUSimulator
from ..isa_mapper.types import ActType, HardwareConfig, NPUProgram
from . import Backend, _run_optimizer


class NPUBackend(Backend):
    """Compiles IR graphs to NPU hardware instructions.

    Usage::

        backend = NPUBackend()
        result = backend.run(graph, {"x": input_data})

    Limitations:
        - Only matmul-based ops are hardware-accelerated.
        - conv2d, pool, softmax fall back to NumPy.
        - K-tiling (K > 4) not yet supported by ISA Mapper.
    """

    def __init__(self, config: HardwareConfig | None = None) -> None:
        self.config = config or HardwareConfig()
        self.mapper = ISAMapper(config=self.config)
        self._last_program: NPUProgram | None = None
        self._last_source: str = ""

    # ── Public API ───────────────────────────────────────────────

    def compile(
        self,
        output: Node,
        *,
        graph: Graph | None = None,
        optimize: bool = True,
    ) -> tuple[Callable[..., Any], str, Node, list[str]]:
        """Compile a graph to an NPU-executable callable.

        Returns:
            (callable, source=str(program), optimized_output, logs)
        """
        logs: list[str] = []

        if optimize and graph is not None:
            opt_output, opt_logs = _run_optimizer(graph)
            output = opt_output
            logs.extend(opt_logs)

        # Generate NPU program if the graph is a simple matmul+act pattern
        program = self._lower_to_npu(output, logs)

        source = program.display() if program else "// NPU: no matching pattern"
        self._last_program = program
        self._last_source = source

        def compiled_fn(**inputs: np.ndarray) -> np.ndarray:
            if program is None:
                # Fallback: NumPy reference
                return self._fallback_numpy(output, inputs)
            return self._execute_on_simulator(program, output, inputs)

        return compiled_fn, source, output, logs

    def run(
        self,
        graph: Graph,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Optimize, lower to NPU, simulate, return results."""
        output, logs = _run_optimizer(graph)

        program = self._lower_to_npu(output, logs)
        if program is None:
            raise RuntimeError(
                "Graph cannot be lowered to NPU ISA. "
                "Only matmul + add + relu patterns are supported."
            )

        sim = NPUSimulator(self.config)
        sim.load_program(program)

        # Pack inputs into external memory
        ext_mem: dict[int, int] = {}
        offset = 0
        input_nodes = [n for n in graph.nodes
                       if n.op == "input" and n.name in inputs]
        for node in input_nodes:
            arr = inputs[node.name]
            flat = np.asarray(arr, dtype=np.int32).ravel()
            for i, v in enumerate(flat):
                ext_mem[offset + i] = int(v)
            offset += max(len(flat), 64)  # spacing between inputs

        sim.load_external_memory(ext_mem)
        sim.run()

        # Read output from ext_mem[128+]
        c_size = 0
        for instr in program.instructions:
            if instr._opcode_int == 0x2:
                c_size = (instr.binary >> 16) & 0x3FF
                break
        if c_size == 0:
            c_size = 16

        results = np.array(sim.read_external_memory(128, c_size), dtype=np.int32)
        return {"output": results}

    # ── Internal ─────────────────────────────────────────────────

    def _lower_to_npu(
        self,
        output: Node,
        logs: list[str],
    ) -> NPUProgram | None:
        """Attempt to lower a graph to an NPU program.

        Only handles: matmul + add + relu (or fused_mma_bias_relu).
        """
        if output.op == "fused_mma_bias_relu":
            x, w, b = output.inputs
            return self._lower_fused_mma(x, w, b, logs)

        # Try: relu(add(matmul(x, w), b))
        if output.op == "relu" and output.inputs[0].op == "add":
            add_node = output.inputs[0]
            left, right = add_node.inputs
            if left.op == "matmul":
                x, w = left.inputs
                b = right
                return self._lower_fused_mma(x, w, b, logs, act=ActType.RELU)
            if right.op == "matmul":
                x, w = right.inputs
                b = left
                return self._lower_fused_mma(x, w, b, logs, act=ActType.RELU)

        # Try: add(matmul(x, w), b)
        if output.op == "add":
            left, right = output.inputs
            if left.op == "matmul":
                x, w = left.inputs
                b = right
                return self._lower_fused_mma(x, w, b, logs)
            if right.op == "matmul":
                x, w = right.inputs
                b = left
                return self._lower_fused_mma(x, w, b, logs)

        # Pure matmul
        if output.op == "matmul":
            x, w = output.inputs
            return self._lower_fused_mma(x, w, None, logs)

        logs.append("[npu] cannot lower — unsupported op pattern")
        return None

    def _lower_fused_mma(
        self,
        x: Node,
        w: Node,
        b: Node | None,
        logs: list[str],
        act: ActType = ActType.NONE,
    ) -> NPUProgram | None:
        """Lower a matmul(+bias)(+relu) to NPU program."""
        if not (isinstance(w.value, np.ndarray) and w.op == "const"):
            logs.append("[npu] weight must be const for NPU lowering")
            return None

        w_arr = w.value
        if w_arr.ndim != 2:
            logs.append("[npu] weight must be 2D")
            return None

        K_w, N = w_arr.shape
        logs.append(f"[npu] lowering matmul: W({K_w}×{N}), act={act.name}")

        return self.mapper.map_matmul(
            M=4,  # batch size — infer from input later
            N=N,
            K=K_w,
            ext_a_addr=0,
            ext_b_addr=64,
            ext_c_addr=128,
            act=act,
        )

    @staticmethod
    def _execute_on_simulator(
        program: NPUProgram,
        output: Node,
        inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Execute a program on the NPU simulator and return results."""
        sim = NPUSimulator()
        sim.load_program(program)

        ext_mem: dict[int, int] = {}
        offset = 0
        for name, arr in inputs.items():
            flat = np.asarray(arr, dtype=np.int32).ravel()
            for i, v in enumerate(flat):
                ext_mem[offset + i] = int(v)
            offset += max(len(flat), 64)

        sim.load_external_memory(ext_mem)
        sim.run()

        results = sim.read_external_memory(128, 16)
        return np.array(results, dtype=np.int32)

    @staticmethod
    def _fallback_numpy(
        output: Node,
        inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Fallback to NumPy reference execution."""
        from ..codegen import NumpyCompiler
        compiler = NumpyCompiler()
        fn, _, _, _ = compiler.compile(output, optimize=False)
        kwargs = {k: v for k, v in inputs.items()}
        return fn(**kwargs)
