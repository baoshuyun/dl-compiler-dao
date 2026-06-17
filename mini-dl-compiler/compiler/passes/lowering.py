"""Graph IR → SSA IR lowering pass.

Converts the high-level Graph IR (Node-based) into the lower-level
SSA IR (Operation/Value-based) for further lowering to target code.

This is the bridge between the graph optimization domain (fold, fuse,
DCE) and the SSA-based optimization domain (tiling scheduling,
memory bufferization).
"""

from __future__ import annotations

from typing import Any

from ..ir import Graph, Node
from ..ssair import (
    MemRefType,
    Operation,
    SSAValue,
    TensorType,
    make_memref_value,
    make_tensor_value,
)
from . import Pass


class GraphToSSAPass(Pass):
    """Lower a Graph IR into an SSA IR module.

    Each Node becomes one or more SSA Operations.  Input/Const nodes
    produce SSA Values; compute nodes produce Operations that consume
    input Values and produce result Values.

    The output is a list of SSA Operations (a flat module).

    Attributes:
        bufferize: If True, convert tensor types to memref types
                   (explicit memory — needed before hardware codegen).
    """

    name = "Graph → SSA Lowering"

    def __init__(self, bufferize: bool = True) -> None:
        self.bufferize = bufferize

    def run(self, ir: Graph, **kwargs: Any) -> list[Operation]:
        """Lower a Graph into an SSA module.

        Returns:
            List of SSA Operations in topological order.
        """
        if ir.output is None:
            return []

        try:
            order = ir.topological_order(ir.output)
        except RecursionError:
            return []

        node_to_value: dict[int, SSAValue] = {}
        ops: list[Operation] = []

        for node in order:
            if node.op == "input":
                val = self._emit_input(node)
                node_to_value[id(node)] = val

            elif node.op == "const":
                val = self._emit_const(node)
                node_to_value[id(node)] = val
                ops.append(val.defining_op)  # type: ignore[union-attr]

            elif node.op == "add":
                op = self._emit_binary("arith.addf", node, node_to_value)
                ops.append(op)
                node_to_value[id(node)] = op.results[0]

            elif node.op == "matmul":
                op = self._emit_matmul(node, node_to_value)
                ops.append(op)
                node_to_value[id(node)] = op.results[0]

            elif node.op == "fused_mma_bias_relu":
                op = self._emit_fused_mma(node, node_to_value)
                ops.append(op)
                node_to_value[id(node)] = op.results[0]

            elif node.op == "relu":
                op = self._emit_unary("linalg.relu", node, node_to_value)
                ops.append(op)
                node_to_value[id(node)] = op.results[0]

            elif node.op == "conv2d":
                op = self._emit_conv2d(node, node_to_value)
                ops.append(op)
                node_to_value[id(node)] = op.results[0]

            elif node.op == "reshape":
                op = self._emit_reshape(node, node_to_value)
                ops.append(op)
                node_to_value[id(node)] = op.results[0]

            elif node.op == "softmax":
                op = self._emit_unary("linalg.softmax", node, node_to_value)
                ops.append(op)
                node_to_value[id(node)] = op.results[0]

            else:
                # Passthrough: emit as generic op
                op = self._emit_generic(node, node_to_value)
                if op:
                    ops.append(op)
                    node_to_value[id(node)] = op.results[0]

        return ops

    def verify(self, before: Graph, after: list[Operation]) -> list[str]:
        warnings = []
        if not after:
            warnings.append("GraphToSSAPass produced no operations")
        return warnings

    # ── Emitters ────────────────────────────────────────────────

    def _emit_input(self, node: Node) -> SSAValue:
        shape = getattr(node, "attrs", {}).get("shape", ())
        if self.bufferize:
            return make_memref_value(node.name, shape)
        return make_tensor_value(node.name, shape)

    def _emit_const(self, node: Node) -> SSAValue:
        import numpy as np
        arr = np.asarray(node.value)
        shape = tuple(arr.shape)
        val = make_tensor_value(node.name or "const", shape)
        op = Operation(
            op_type="arith.constant",
            inputs=[],
            results=[val],
            attributes={"value": arr},
        )
        return val

    def _emit_binary(
        self, op_type: str, node: Node,
        node_to_value: dict[int, SSAValue],
    ) -> Operation:
        in_vals = [node_to_value[id(inp)] for inp in node.inputs]
        out = make_tensor_value(
            node.name or f"{op_type}_out",
            self._result_shape(node),
        )
        return Operation(op_type=op_type, inputs=in_vals, results=[out])

    def _emit_unary(
        self, op_type: str, node: Node,
        node_to_value: dict[int, SSAValue],
    ) -> Operation:
        in_val = node_to_value[id(node.inputs[0])]
        out = make_tensor_value(
            node.name or f"{op_type}_out",
            self._result_shape(node),
        )
        return Operation(op_type=op_type, inputs=[in_val], results=[out])

    def _emit_matmul(
        self, node: Node,
        node_to_value: dict[int, SSAValue],
    ) -> Operation:
        a = node_to_value[id(node.inputs[0])]
        b = node_to_value[id(node.inputs[1])]
        shape = self._result_shape(node)
        out = make_tensor_value(node.name or "matmul_out", shape)
        return Operation(
            op_type="linalg.matmul",
            inputs=[a, b],
            results=[out],
            attributes=dict(node.attrs),
        )

    def _emit_fused_mma(
        self, node: Node,
        node_to_value: dict[int, SSAValue],
    ) -> Operation:
        x, w, b = [node_to_value[id(inp)] for inp in node.inputs]
        shape = self._result_shape(node)
        out = make_tensor_value(node.name or "fused_out", shape)
        return Operation(
            op_type="linalg.fused_mma_bias_relu",
            inputs=[x, w, b],
            results=[out],
            attributes=dict(node.attrs),
        )

    def _emit_conv2d(
        self, node: Node,
        node_to_value: dict[int, SSAValue],
    ) -> Operation:
        x = node_to_value[id(node.inputs[0])]
        w = node_to_value[id(node.inputs[1])]
        shape = self._result_shape(node)
        out = make_tensor_value(node.name or "conv2d_out", shape)
        return Operation(
            op_type="linalg.conv_2d_nhwc_hwcf",
            inputs=[x, w],
            results=[out],
            attributes=dict(node.attrs),
        )

    def _emit_reshape(
        self, node: Node,
        node_to_value: dict[int, SSAValue],
    ) -> Operation:
        x = node_to_value[id(node.inputs[0])]
        new_shape = node.attrs.get("shape", ())
        out = make_tensor_value(node.name or "reshape_out", new_shape)
        return Operation(
            op_type="tensor.reshape",
            inputs=[x],
            results=[out],
            attributes={"shape": new_shape},
        )

    def _emit_generic(
        self, node: Node,
        node_to_value: dict[int, SSAValue],
    ) -> Operation | None:
        """Emit a generic operation for unknown op types."""
        in_vals = [node_to_value[id(inp)] for inp in node.inputs]
        if not in_vals and node.op not in ("input", "const"):
            return None
        out = make_tensor_value(node.name or f"{node.op}_out", ())
        return Operation(
            op_type=f"unknown.{node.op}",
            inputs=in_vals,
            results=[out],
            attributes=dict(node.attrs),
        )

    # ── Shape helpers ───────────────────────────────────────────

    @staticmethod
    def _result_shape(node: Node) -> tuple[int, ...]:
        """Infer the output shape of a node."""
        import numpy as np
        if isinstance(node.value, np.ndarray):
            return tuple(node.value.shape)
        if "shape" in node.attrs:
            return tuple(node.attrs["shape"])
        # Try to infer from inputs
        for inp in node.inputs:
            if isinstance(inp.value, np.ndarray):
                return tuple(inp.value.shape)
        return ()
