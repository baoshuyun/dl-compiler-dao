"""ONNX Frontend — lowers ONNX models to the linalg dialect.

Parses ONNX model files (``.onnx``) and extracts the computational
graph, mapping ONNX operators to linalg operations for downstream
lowering through the NPU or RVV backends.

Usage::

    lowerer = OnnxToLinalg()
    program = lowerer.lower("resnet18.onnx")
    for m in program.matmuls:
        print(m)

Supported ONNX ops:
    MatMul, Gemm, Conv, Relu, Add, Softmax, Reshape,
    BatchNormalization, MaxPool, GlobalAveragePool
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .lowering_pipeline import LinalgMatmul
from .npu_dialect import ActType


# ════════════════════════════════════════════════════════════════
# ONNX → Linalg data types
# ════════════════════════════════════════════════════════════════

@dataclass
class OnnxTensor:
    """A tensor in the ONNX graph.

    Attributes:
        name: Tensor name from the ONNX graph.
        shape: Static shape tuple (``-1`` for dynamic dims).
        dtype: Element data type.
        data: Concrete array for initializers (weights/bias), ``None``
              for activations.
    """
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype = np.dtype("float32")
    data: Optional[np.ndarray] = None


@dataclass
class LinalgConv2D:
    """linalg.conv_2d_nhwc_hwcf operation.

    NHWC input layout, HWCF filter layout — matches the ONNX Conv
    convention after a trivial NCHW→NHWC transpose.

    Attributes:
        input_name: SSA name for the input feature map.
        filter_name: SSA name for the convolution kernel.
        output_name: SSA name for the output feature map.
        n: Batch size.
        h, w: Input spatial dimensions.
        c: Input channel count.
        r, s: Filter spatial dimensions.
        c_out: Output channel count.
        stride: Convolution stride (H, W).
        padding: Padding (H, W).
        act: Optional fused activation.
    """
    input_name: str
    filter_name: str
    output_name: str
    n: int = 1
    h: int = 1
    w: int = 1
    c: int = 1
    r: int = 1
    s: int = 1
    c_out: int = 1
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    act: ActType = ActType.NONE

    def __repr__(self) -> str:
        return (
            f"linalg.conv_2d {self.output_name}"
            f"[{self.n},{self.h},{self.w},{self.c_out}] "
            f"= conv({self.input_name}, {self.filter_name})"
        )


@dataclass
class OnnxProgram:
    """Result of lowering an ONNX model to the linalg dialect.

    Attributes:
        matmuls: Extracted linalg.matmul operations.
        convs: Extracted linalg.conv_2d operations.
        activations: Mapping from output tensor name to activation type.
        tensors: All known tensors (inputs, weights, intermediates).
        input_names: Model input tensor names.
        output_names: Model output tensor names.
    """
    matmuls: list[LinalgMatmul] = field(default_factory=list)
    convs: list[LinalgConv2D] = field(default_factory=list)
    activations: dict[str, ActType] = field(default_factory=dict)
    tensors: dict[str, OnnxTensor] = field(default_factory=dict)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)

    # ── Derived properties ─────────────────────────────────────

    @property
    def has_matmul(self) -> bool:
        """True if the model contains at least one matmul."""
        return len(self.matmuls) > 0

    @property
    def has_conv(self) -> bool:
        """True if the model contains at least one convolution."""
        return len(self.convs) > 0

    @property
    def primary_op(self) -> str:
        """Return the primary compute operation type."""
        if self.convs:
            return "conv2d"
        if self.matmuls:
            return "matmul"
        return "unknown"

    @property
    def total_ops(self) -> int:
        """Total number of compute operations."""
        return len(self.matmuls) + len(self.convs)

    def __repr__(self) -> str:
        parts = []
        if self.matmuls:
            parts.append(f"{len(self.matmuls)} matmuls")
        if self.convs:
            parts.append(f"{len(self.convs)} convs")
        acts = sum(1 for a in self.activations.values() if a != ActType.NONE)
        if acts:
            parts.append(f"{acts} activations")
        return f"OnnxProgram({', '.join(parts) or 'empty'})"


# ════════════════════════════════════════════════════════════════
# ONNX → Linalg lowering
# ════════════════════════════════════════════════════════════════

class OnnxToLinalg:
    """Lower an ONNX model to the linalg dialect.

    Usage::

        lowerer = OnnxToLinalg()
        program = lowerer.lower("model.onnx")
        for m in program.matmuls:
            pipeline = NPULoweringPipeline()
            npu_prog, _ = pipeline.lower(m)
    """

    # ── Public API ──────────────────────────────────────────────

    def lower(self, model_path: str) -> OnnxProgram:
        """Parse an ONNX model file and lower to linalg.

        Args:
            model_path: Filesystem path to a ``.onnx`` file.

        Returns:
            ``OnnxProgram`` with extracted linalg operations.

        Raises:
            ImportError: If the ``onnx`` package is not installed.
            FileNotFoundError: If *model_path* does not exist.
        """
        import onnx

        model = onnx.load(model_path)
        return self._lower_model(model)

    def lower_from_bytes(self, model_bytes: bytes) -> OnnxProgram:
        """Parse an ONNX model from a byte buffer.

        Args:
            model_bytes: Serialized ONNX ``ModelProto``.

        Returns:
            ``OnnxProgram`` with extracted linalg operations.
        """
        import onnx

        model = onnx.load_from_string(model_bytes)
        return self._lower_model(model)

    # ── Internal: model lowering ─────────────────────────────────

    def _lower_model(self, model: Any) -> OnnxProgram:
        """Lower an ONNX ``ModelProto`` to an ``OnnxProgram``.

        Processing order:
            1. Extract initializers (weights).
            2. Extract input/output metadata.
            3. Build value_info for intermediate shapes.
            4. Lower each node in topological order.
            5. Back-propagate tracked activations to compute ops.
        """
        import onnx
        from onnx import numpy_helper

        program = OnnxProgram()

        # Step 1: initializers → OnnxTensor with data
        initializers: dict[str, OnnxTensor] = {}
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            t = OnnxTensor(
                name=init.name,
                shape=tuple(arr.shape),
                dtype=arr.dtype,
                data=arr,
            )
            initializers[init.name] = t
            program.tensors[init.name] = t

        # Step 2: inputs / outputs
        for inp in model.graph.input:
            shape = _proto_shape(inp)
            if inp.name not in initializers:
                program.input_names.append(inp.name)
                program.tensors[inp.name] = OnnxTensor(
                    name=inp.name, shape=shape,
                )

        for out in model.graph.output:
            shape = _proto_shape(out)
            program.output_names.append(out.name)
            program.tensors[out.name] = OnnxTensor(
                name=out.name, shape=shape,
            )

        # Step 3: value_info shape cache (seed with input shapes)
        value_shapes: dict[str, tuple[int, ...]] = {}
        for inp in model.graph.input:
            if inp.name not in initializers:
                value_shapes[inp.name] = _proto_shape(inp)
        for vi in model.graph.value_info:
            value_shapes[vi.name] = _proto_shape(vi)

        # Step 4: node-by-node lowering
        tracked_act: dict[str, Optional[str]] = {}

        for node in model.graph.node:
            if node.op_type == "MatMul":
                m = self._lower_matmul(node, initializers, value_shapes)
                program.matmuls.append(m)
                tracked_act[m.result] = None

            elif node.op_type == "Gemm":
                m = self._lower_gemm(node, initializers, value_shapes)
                program.matmuls.append(m)
                tracked_act[m.result] = None

            elif node.op_type == "Conv":
                c = self._lower_conv(node, initializers, value_shapes)
                program.convs.append(c)
                tracked_act[c.output_name] = None

            elif node.op_type == "Relu":
                inp_name = node.input[0]
                out_name = node.output[0]
                for prev_out in tracked_act:
                    if prev_out == inp_name:
                        tracked_act[prev_out] = "relu"
                program.activations[out_name] = ActType.RELU
                value_shapes[out_name] = value_shapes.get(inp_name, ())

            elif node.op_type == "Add":
                value_shapes[node.output[0]] = value_shapes.get(
                    node.input[0], (),
                )

            elif node.op_type == "Softmax":
                value_shapes[node.output[0]] = value_shapes.get(
                    node.input[0], (),
                )

            elif node.op_type == "Reshape":
                shape_in = node.input[1] if len(node.input) > 1 else ""
                if shape_in in initializers and initializers[shape_in].data is not None:
                    new_shape = tuple(
                        int(x) for x in initializers[shape_in].data
                    )
                else:
                    new_shape = value_shapes.get(node.input[0], ())
                value_shapes[node.output[0]] = new_shape

            elif node.op_type in (
                "BatchNormalization", "MaxPool", "GlobalAveragePool",
            ):
                value_shapes[node.output[0]] = value_shapes.get(
                    node.input[0], (),
                )

        # Step 5: back-propagate activations to compute ops
        for matmul in program.matmuls:
            if tracked_act.get(matmul.result) == "relu":
                matmul.act = ActType.RELU

        for conv in program.convs:
            if tracked_act.get(conv.output_name) == "relu":
                conv.act = ActType.RELU

        return program

    # ── Op-specific lowerers ─────────────────────────────────────

    def _lower_matmul(
        self,
        node: Any,
        initializers: dict[str, OnnxTensor],
        shapes: dict[str, tuple[int, ...]],
    ) -> LinalgMatmul:
        """Lower ONNX ``MatMul`` to ``linalg.matmul``.

        ONNX MatMul operates on the last two dimensions; leading
        dimensions are treated as a batch and broadcast.
        """
        a_name = node.input[0]
        b_name = node.input[1]
        out_name = node.output[0]

        a_shape = _resolve_shape(a_name, initializers, shapes)
        b_shape = _resolve_shape(b_name, initializers, shapes)

        if len(a_shape) >= 2 and len(b_shape) >= 2:
            M, K = a_shape[-2], a_shape[-1]
            K2, N = b_shape[-2], b_shape[-1]
        else:
            M = K = N = 4

        return LinalgMatmul(
            lhs=a_name, rhs=b_name, result=out_name,
            m=M, n=N, k=K,
        )

    def _lower_gemm(
        self,
        node: Any,
        initializers: dict[str, OnnxTensor],
        shapes: dict[str, tuple[int, ...]],
    ) -> LinalgMatmul:
        """Lower ONNX ``Gemm`` (Y = α·A·B + β·C) to ``linalg.matmul``.

        Handles optional transposition via ``transA`` / ``transB``.
        """
        a_name = node.input[0]
        b_name = node.input[1]
        out_name = node.output[0]

        a_shape = _resolve_shape(a_name, initializers, shapes)
        b_shape = _resolve_shape(b_name, initializers, shapes)

        trans_a = _get_attr(node, "transA", 0)
        trans_b = _get_attr(node, "transB", 0)

        if len(a_shape) >= 2 and len(b_shape) >= 2:
            if trans_a:
                a_shape = (a_shape[1], a_shape[0])
            if trans_b:
                b_shape = (b_shape[1], b_shape[0])
            M, K = a_shape[-2], a_shape[-1]
            _, N = b_shape[-2], b_shape[-1]
        else:
            M = K = N = 4

        return LinalgMatmul(
            lhs=a_name, rhs=b_name, result=out_name,
            m=M, n=N, k=K,
        )

    def _lower_conv(
        self,
        node: Any,
        initializers: dict[str, OnnxTensor],
        shapes: dict[str, tuple[int, ...]],
    ) -> LinalgConv2D:
        """Lower ONNX ``Conv`` to ``linalg.conv_2d_nhwc_hwcf``.

        ONNX uses NCHW layout; this method records the shapes and
        leaves the layout transform to a later pass.
        """
        x_name = node.input[0]
        w_name = node.input[1]
        out_name = node.output[0]

        x_shape = _resolve_shape(x_name, initializers, shapes)
        w_shape = _resolve_shape(w_name, initializers, shapes)

        if len(x_shape) == 4 and len(w_shape) == 4:
            n, c_in, h, w = x_shape
            c_out, _, r, s = w_shape
            stride = _get_attr(node, "strides", [1, 1])
            pads = _get_attr(node, "pads", [0, 0, 0, 0])
        else:
            n = c_in = h = w = 1
            c_out = r = s = 1
            stride = (1, 1)
            pads = (0, 0)

        return LinalgConv2D(
            input_name=x_name, filter_name=w_name, output_name=out_name,
            n=n, h=h, w=w, c=c_in,
            r=r, s=s, c_out=c_out,
            stride=tuple(stride[:2]),
            padding=tuple(pads[:2]),
        )


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _proto_shape(tensor_proto: Any) -> tuple[int, ...]:
    """Extract shape from an ONNX ``ValueInfoProto`` or ``TensorProto``.

    Returns an empty tuple if the type information is missing.
    """
    if not hasattr(tensor_proto, "type"):
        return ()
    tp = tensor_proto.type
    if hasattr(tp, "tensor_type") and hasattr(tp.tensor_type, "shape"):
        dims = tp.tensor_type.shape.dim
        if dims:
            return tuple(
                d.dim_value if d.dim_value > 0 else -1
                for d in dims
            )
    return ()


def _resolve_shape(
    name: str,
    initializers: dict[str, OnnxTensor],
    shapes: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    """Resolve a tensor's shape from initializers or value_info.

    Args:
        name: Tensor name.
        initializers: ONNX initializer tensors (weights).
        shapes: Value-info shape cache.

    Returns:
        Resolved shape tuple, or ``()`` if unknown.
    """
    if name in initializers:
        return initializers[name].shape
    return shapes.get(name, ())


def _get_attr(node: Any, name: str, default: Any = None) -> Any:
    """Extract a named attribute from an ONNX ``NodeProto``.

    Handles ``INTS`` (list), ``INT`` (scalar), and ``FLOAT`` types.
    """
    for attr in node.attribute:
        if attr.name == name:
            # ONNX AttributeType enum values
            if hasattr(attr, "ints") and attr.ints:
                return list(attr.ints)
            if hasattr(attr, "i"):
                return attr.i
            if hasattr(attr, "f"):
                return attr.f
    return default


# ════════════════════════════════════════════════════════════════
# Self-test
# ════════════════════════════════════════════════════════════════

def _test() -> None:
    """Build a minimal MatMul + ReLU ONNX model and verify lowering."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        print("[SKIP] onnx package not installed")
        return

    matmul_node = helper.make_node(
        "MatMul", inputs=["A", "B"], outputs=["C_pre"],
    )
    relu_node = helper.make_node(
        "Relu", inputs=["C_pre"], outputs=["C"],
    )
    graph = helper.make_graph(
        [matmul_node, relu_node],
        "test_graph",
        inputs=[
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 4]),
        ],
    )
    model = helper.make_model(graph)
    model_bytes = model.SerializeToString()

    lowerer = OnnxToLinalg()
    program = lowerer.lower_from_bytes(model_bytes)

    assert len(program.matmuls) == 1, (
        f"Expected 1 matmul, got {len(program.matmuls)}"
    )
    m = program.matmuls[0]
    assert m.m == 4 and m.n == 4 and m.k == 8, (
        f"Wrong matmul shape: {m}"
    )
    assert m.act == ActType.RELU, (
        "ReLU activation not tracked"
    )
    print(f"[PASS] ONNX Frontend: {m}")


if __name__ == "__main__":
    _test()
