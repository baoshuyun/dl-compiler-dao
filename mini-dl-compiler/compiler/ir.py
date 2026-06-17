"""Core IR (Intermediate Representation) for the DL compiler.

Defines the computation graph and node types used throughout the compiler pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class Node:
    """A node in the computation graph.

    Nodes use identity-based hashing/equality (not value-based) so that distinct
    nodes with the same op and value are treated as different graph vertices.
    """

    __slots__ = ("attrs", "inputs", "name", "op", "value")

    def __init__(
        self,
        op: str,
        inputs: Sequence[Node] | None = None,
        value: Any = None,
        name: str = "",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.op = op
        self.inputs: list[Node] = list(inputs) if inputs else []
        self.value: Any = value
        self.name = name
        self.attrs: dict[str, Any] = dict(attrs) if attrs else {}

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        label = f"'{self.name}'" if self.name else "-"
        extra = ""
        if self.op == "const" and isinstance(self.value, np.ndarray):
            extra = f" shape={self.value.shape}"
        return f"Node(op={self.op!r}, name={label}{extra})"


class Graph:
    """A computation graph that owns all nodes.

    All node creation should go through the factory methods on this class
    so that the nodes list stays consistent.
    """

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.output: Node | None = None

    # ------------------------------------------------------------------
    # Node factories
    # ------------------------------------------------------------------

    def _add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def Input(self, name: str) -> Node:
        if not name:
            raise ValueError("Input node must have a non-empty name")
        return self._add(Node(op="input", name=name))

    def Const(self, value: Any, name: str = "") -> Node:
        arr = np.asarray(value)
        return self._add(Node(op="const", value=arr, name=name))

    def Add(self, a: Node, b: Node, name: str = "") -> Node:
        self._check_inputs(a, b)
        return self._add(Node(op="add", inputs=[a, b], name=name))

    def MatMul(self, a: Node, b: Node, name: str = "") -> Node:
        self._check_inputs(a, b)
        return self._add(Node(op="matmul", inputs=[a, b], name=name))

    def ReLU(self, a: Node, name: str = "") -> Node:
        self._check_inputs(a)
        return self._add(Node(op="relu", inputs=[a], name=name))

    def FusedMMA_Bias_ReLU(self, x: Node, w: Node, b: Node, name: str = "") -> Node:
        """Create a fused matmul + bias + relu node (used by the fusion pass)."""
        self._check_inputs(x, w, b)
        return self._add(Node(op="fused_mma_bias_relu", inputs=[x, w, b], name=name))

    def Conv2D(self, x: Node, w: Node, *,
               strides: tuple[int, int] = (1, 1),
               padding: tuple[int, int] = (0, 0),
               name: str = "") -> Node:
        """2D convolution: x[N,H,W,C_in] @ w[Kh,Kw,C_in,C_out]."""
        self._check_inputs(x, w)
        attrs = {"strides": strides, "padding": padding}
        return self._add(Node(op="conv2d", inputs=[x, w], name=name, attrs=attrs))

    def MaxPool2D(self, x: Node, *,
                  kernel_size: tuple[int, int] = (2, 2),
                  strides: tuple[int, int] | None = None,
                  name: str = "") -> Node:
        """2D max pooling."""
        self._check_inputs(x)
        s = strides if strides is not None else kernel_size
        attrs = {"kernel_size": kernel_size, "strides": s}
        return self._add(Node(op="max_pool2d", inputs=[x], name=name, attrs=attrs))

    def BatchNorm(self, x: Node, scale: Node, bias: Node,
                  mean: Node, var: Node, *,
                  epsilon: float = 1e-5,
                  name: str = "") -> Node:
        """Batch normalization: (x - mean) / sqrt(var + eps) * scale + bias."""
        self._check_inputs(x, scale, bias, mean, var)
        attrs = {"epsilon": epsilon}
        return self._add(Node(op="batch_norm",
                         inputs=[x, scale, bias, mean, var], name=name, attrs=attrs))

    def Softmax(self, x: Node, *, axis: int = -1, name: str = "") -> Node:
        """Softmax activation."""
        self._check_inputs(x)
        return self._add(Node(op="softmax", inputs=[x], name=name, attrs={"axis": axis}))

    def Reshape(self, x: Node, shape: tuple[int, ...], name: str = "") -> Node:
        """Reshape tensor to *shape*."""
        self._check_inputs(x)
        return self._add(Node(op="reshape", inputs=[x], name=name, attrs={"shape": shape}))

    def Transpose(self, x: Node, axes: tuple[int, ...] | None = None,
                  name: str = "") -> Node:
        """Permute tensor axes."""
        self._check_inputs(x)
        return self._add(Node(op="transpose", inputs=[x], name=name,
                         attrs={"axes": axes}))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _check_inputs(*nodes: Node) -> None:
        for n in nodes:
            if not isinstance(n, Node):
                raise TypeError(f"Expected Node, got {type(n).__name__}: {n!r}")

    def validate(self) -> list[str]:
        """Run sanity checks on the graph. Returns a list of warnings."""
        warnings: list[str] = []
        node_set = set(self.nodes)

        if self.output is not None and self.output not in node_set:
            warnings.append("Graph.output is not in graph.nodes")

        for node in self.nodes:
            for inp in node.inputs:
                if inp not in node_set:
                    warnings.append(
                        f"Node {node!r} references input {inp!r} "
                        f"which is not in graph.nodes"
                    )
        return warnings

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------

    def reachable_from(self, root: Node) -> set[Node]:
        """Return the set of nodes reachable from *root* (including *root*)."""
        seen: set[Node] = set()

        def visit(n: Node) -> None:
            if n in seen:
                return
            seen.add(n)
            for ch in n.inputs:
                visit(ch)

        visit(root)
        return seen

    def topological_order(self, root: Node) -> list[Node]:
        """Return nodes reachable from *root* in topological order (inputs first)."""
        order: list[Node] = []
        perm: set[Node] = set()
        temp: set[Node] = set()

        def visit(n: Node) -> None:
            if n in perm:
                return
            if n in temp:
                raise RecursionError("Graph contains a cycle")
            temp.add(n)
            for ch in n.inputs:
                visit(ch)
            temp.discard(n)
            perm.add(n)
            order.append(n)

        visit(root)
        return order
