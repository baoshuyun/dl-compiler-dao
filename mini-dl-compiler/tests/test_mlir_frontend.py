"""Tests for MLIR frontend and ONNX lowering."""
from __future__ import annotations

import numpy as np

from compiler.mlir.frontend import TaskGraphToLinalg
from compiler.mlir.onnx_frontend import OnnxToLinalg, OnnxProgram
from compiler.mlir.rvv_lowering import RVVLoweringPipeline
from compiler.mlir.lowering_pipeline import LinalgMatmul
from compiler.mlir.npu_dialect import ActType


class _FakeTaskGraph:
    """Minimal fake TaskGraph for testing TaskGraphToLinalg."""
    def __init__(self, tasks):
        self.tasks = tasks


class _FakeTask:
    def __init__(self, op, inputs, output, **kw):
        self.op = op
        self.inputs = inputs
        self.output = output
        for k, v in kw.items():
            setattr(self, k, v)


class TestTaskGraphToLinalg:
    """Frontend: Soft_Stack TaskGraph → linalg bridge."""

    def test_lower_simple_matmul(self) -> None:
        lowerer = TaskGraphToLinalg()
        task = _FakeTask("matmul", ["A", "B"], "C")
        graph = _FakeTaskGraph([task])
        inputs = {"A": np.ones((4, 8)), "B": np.ones((8, 4))}
        prog = lowerer.lower(graph, inputs)
        assert prog.matmul.m == 4
        assert prog.matmul.n == 4
        assert prog.matmul.k == 8

    def test_lower_matmul_relu(self) -> None:
        lowerer = TaskGraphToLinalg()
        t1 = _FakeTask("matmul", ["A", "B"], "D")
        t2 = _FakeTask("relu", ["D"], "C")
        graph = _FakeTaskGraph([t1, t2])
        inputs = {"A": np.ones((4, 8)), "B": np.ones((8, 4))}
        prog = lowerer.lower(graph, inputs)
        assert prog.act == ActType.RELU

    def test_lower_mul_no_relu(self) -> None:
        lowerer = TaskGraphToLinalg()
        task = _FakeTask("mul", ["A", "B"], "C")
        graph = _FakeTaskGraph([task])
        inputs = {"A": np.ones((8, 8)), "B": np.ones((8, 8))}
        prog = lowerer.lower(graph, inputs)
        assert prog.matmul.k == 8

    def test_lower_raises_on_empty_graph(self) -> None:
        lowerer = TaskGraphToLinalg()
        graph = _FakeTaskGraph([])
        import pytest
        with pytest.raises(ValueError):
            lowerer.lower(graph, {"A": np.ones((4, 4))})

    def test_lower_raises_on_non_2d(self) -> None:
        lowerer = TaskGraphToLinalg()
        task = _FakeTask("matmul", ["A", "B"], "C")
        graph = _FakeTaskGraph([task])
        import pytest
        with pytest.raises(ValueError):
            lowerer.lower(graph, {"A": np.ones(4), "B": np.ones(4)})


class TestOnnxToLinalg:
    """ONNX frontend: ONNX model → linalg."""

    def test_lower_from_bytes_matmul_relu(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        mm = helper.make_node("MatMul", ["A", "B"], ["C_pre"])
        relu = helper.make_node("Relu", ["C_pre"], ["C"])
        graph = helper.make_graph(
            [mm, relu], "g",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8]),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 4])],
            [helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 4])],
        )
        model = helper.make_model(graph)
        prog = OnnxToLinalg().lower_from_bytes(model.SerializeToString())
        assert len(prog.matmuls) == 1
        assert prog.matmuls[0].act == ActType.RELU

    def test_lower_gemm(self) -> None:
        import onnx
        from onnx import TensorProto, helper
        import numpy as onp

        C_arr = onp.zeros((4, 4), dtype=onp.float32)
        C_init = helper.make_tensor("C", TensorProto.FLOAT, [4, 4], C_arr.flatten().tolist())
        gemm = helper.make_node("Gemm", ["A", "B", "C"], ["Y"])
        graph = helper.make_graph(
            [gemm], "g",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8]),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 4])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])],
            [C_init],
        )
        model = helper.make_model(graph)
        prog = OnnxToLinalg().lower_from_bytes(model.SerializeToString())
        assert len(prog.matmuls) == 1


class TestRVVLowering:
    """RVV lowering: linalg → RISC-V Vector assembly."""

    def test_lower_4x4_no_act(self) -> None:
        pipeline = RVVLoweringPipeline()
        asm = pipeline.lower(LinalgMatmul("A", "B", "C", 4, 4, 4))
        assert "vsetvli" in asm
        assert "vle32.v" in asm
        assert "vfmacc.vv" in asm
        assert "vse32.v" in asm
        assert "vfmax.vv" not in asm  # no ReLU

    def test_lower_4x4_relu(self) -> None:
        pipeline = RVVLoweringPipeline()
        asm = pipeline.lower(LinalgMatmul("A", "B", "C", 4, 4, 4), ActType.RELU)
        assert "vfmax.vv" in asm
        assert "vmv.v.i" in asm

    def test_lower_2x2_matmul(self) -> None:
        pipeline = RVVLoweringPipeline()
        asm = pipeline.lower(LinalgMatmul("A", "B", "C", 2, 2, 2))
        assert "vfmacc.vv" in asm
