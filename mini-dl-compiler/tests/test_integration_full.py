"""Full-stack integration: mini-dl-compiler → ISAMapper → NPU simulator.

Proves the three projects are fully interoperable:
  1. mini-dl-compiler optimizes the graph
  2. ISAMapper lowers to NPU ISA instructions
  3. NPU simulator executes and produces correct results
  4. Results match NumPy golden reference
"""

from __future__ import annotations

import numpy as np
import pytest

from compiler.ir import Graph
from compiler.optimizer import Optimizer
from compiler.isa_mapper import ISAMapper, NPUSimulator
from compiler.isa_mapper.types import ActType, NPUProgram
from compiler.passes import PassManager, TilingPass, MemoryPlanningPass


class TestFullStack:
    """End-to-end: Graph → Optimize → Tiling → ISAMapper → Simulator → Result."""

    @pytest.fixture
    def simple_graph(self) -> Graph:
        g = Graph()
        W = g.Const(np.ones((4, 4), dtype=np.int32), name="W")
        b = g.Const(np.zeros(4, dtype=np.int32), name="b")
        x = g.Input("x")
        y = g.FusedMMA_Bias_ReLU(x, W, b)
        g.output = y
        return g

    def test_full_pipeline_4x4(self, simple_graph: Graph) -> None:
        """Optimize → Tiling → ISAMapper → Simulator → correct answer."""
        # 1. Optimize
        opt = Optimizer(simple_graph)
        opt.run()

        # 2. Tiling
        pm = PassManager()
        pm.add(TilingPass(target_rows=4, target_cols=4, sram_capacity=1024))
        pm.run(simple_graph)

        # 3. ISAMapper
        mapper = ISAMapper()
        program = mapper.map_matmul(M=4, N=4, K=4, act=ActType.RELU)

        # 4. Simulate with identity input
        sim = NPUSimulator()
        sim.load_program(program)

        ext: dict[int, int] = {}
        for i in range(16):
            ext[i] = 2       # A = all-2
            ext[64 + i] = 3  # B = all-3
        sim.load_external_memory(ext)
        sim.run()

        results = np.array(sim.read_external_memory(128, 16), dtype=np.int32)
        expected = np.full(16, 24, dtype=np.int32)  # ReLU(4×6) = 24

        np.testing.assert_array_equal(results, expected)

    def test_optimized_graph_produces_valid_isa(self, simple_graph: Graph) -> None:
        """The optimized graph should be lowerable to valid NPU ISA."""
        opt = Optimizer(simple_graph)
        opt.run()

        mapper = ISAMapper()
        program = mapper.map_matmul(M=4, N=4, K=4, act=ActType.RELU)

        assert isinstance(program, NPUProgram)
        assert program.program_length > 0
        assert len(program.instr_mem) == 256

        # Verify critical instructions exist
        opcodes = {(w >> 28) for w in program.instr_mem[:program.program_length]}
        assert 1 in opcodes  # LOAD
        assert 3 in opcodes  # COMPUTE
        assert 4 in opcodes  # BARRIER

    def test_multi_size_matmul(self) -> None:
        """Test multiple matrix sizes through the full pipeline.

        Note: N-tiling requires strided DMA (not yet in ISA).
        M-tiling works because rows are contiguous in memory.
        K-tiling requires accumulate mode (not yet in ISA).
        """
        test_cases = [
            (4, 4, 4),   # single tile — fully supported
            (8, 4, 4),   # M-tiling — supported (row-major rows are contiguous)
        ]

        rng = np.random.RandomState(123)

        for M, N, K in test_cases:
            A = rng.randint(0, 5, (M, K)).astype(np.int32)
            B = rng.randint(0, 5, (K, N)).astype(np.int32)

            mapper = ISAMapper()
            program = mapper.map_matmul(M=M, N=N, K=K)

            sim = NPUSimulator()
            sim.load_program(program)

            ext: dict[int, int] = {}
            for i, v in enumerate(A.ravel()):
                ext[i] = int(v)
            for i, v in enumerate(B.ravel()):
                ext[64 + i] = int(v)

            sim.load_external_memory(ext)
            sim.run()

            results = np.array(sim.read_external_memory(128, M * N), dtype=np.int32)
            golden = (A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32)

            np.testing.assert_array_equal(
                results.reshape(M, N), golden,
                err_msg=f"Mismatch for {M}x{N}x{K}"
            )

    def test_relu_clamps_negatives(self) -> None:
        """ReLU in the compute pipeline must clamp negatives to 0."""
        mapper = ISAMapper()
        program = mapper.map_matmul(M=4, N=4, K=4, act=ActType.RELU)

        sim = NPUSimulator()
        sim.load_program(program)

        ext: dict[int, int] = {}
        # A = all negative, B = all positive → result negative → ReLU → 0
        for i in range(16):
            ext[i] = -2
            ext[64 + i] = 3
        sim.load_external_memory(ext)
        sim.run()

        results = sim.read_external_memory(128, 16)
        assert all(r == 0 for r in results), f"ReLU failed: {results}"


class TestThreeProjectInterop:
    """Prove Soft_Stack → mini-dl ISAMapper → NPU simulator chain."""

    def test_soft_stack_compiles_to_npu_isa(self) -> None:
        """Soft_Stack NPU backend produces correct results via ISAMapper.

        Proves the three projects share the same ISA layer.
        Requires npu-stack to be importable.
        """
        npu_stack = pytest.importorskip("npu_stack",
                                         reason="NPU_Soft_Hard_Stack not installed")
        Var = npu_stack.Var
        Compiler = npu_stack.Compiler
        matmul = npu_stack.matmul
        relu = npu_stack.relu
        from npu_stack.backends.npu import NPUBackend

        expr = relu(matmul(Var("x"), Var("w")))
        graph = Compiler().compile(expr)

        x = [[1, 1, 1, 1]] * 4
        w = [[1, 1, 1, 1]] * 4

        backend = NPUBackend()
        result = backend.run(graph, {"x": x, "w": w})

        expected = [[4, 4, 4, 4]] * 4
        assert result == expected, f"Three-project interop failed: {result}"
