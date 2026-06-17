"""Optimization passes for the DL compiler IR.

Passes:
  - Constant folding: statically evaluate const-only sub-graphs
  - Operator fusion: merge matmul + add + relu into a single fused op
  - Dead code elimination: remove nodes not reachable from the output
"""

from __future__ import annotations

import numpy as np

from .ir import Graph, Node


class Optimizer:
    """Runs optimization passes on a computation graph.

    Usage::

        opt = Optimizer(graph)
        output = opt.fold(graph.output)
        output = opt.fuse(output)
        result = opt.dce(output)
        print(opt.logs)
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.logs: list[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.logs.append(msg)

    def _copy_meta(self, src: Node, dst: Node) -> None:
        dst.name = src.name
        dst.attrs = dict(src.attrs)

    # ------------------------------------------------------------------
    # Constant folding
    # ------------------------------------------------------------------

    def fold(self, node: Node, memo: dict[Node, Node] | None = None) -> Node:
        """Bottom-up constant folding.

        Returns a new (or existing) node that is the folded equivalent
        of *node*.  Pure-constant sub-graphs are evaluated at compile time.
        """
        if memo is None:
            memo = {}

        if node in memo:
            return memo[node]

        # Leaf nodes don't change.
        if node.op in ("input", "const"):
            memo[node] = node
            return node

        # Recurse on inputs first.
        folded_inputs = [self.fold(inp, memo) for inp in node.inputs]

        out: Node

        if node.op == "add":
            a, b = folded_inputs

            # Identity: x + 0 == 0 + x == x
            if a.op == "const" and np.all(a.value == 0):
                self._log(f"[fold] add(0, x) -> x  ({_label(b)})")
                out = b
            elif b.op == "const" and np.all(b.value == 0):
                self._log(f"[fold] add(x, 0) -> x  ({_label(a)})")
                out = a
            elif a.op == "const" and b.op == "const":
                self._log(
                    f"[fold] const + const -> const  "
                    f"({a.value.shape} + {b.value.shape})"
                )
                out = self.graph.Const(a.value + b.value)
            elif folded_inputs == node.inputs:
                out = node  # no change — reuse the original
            else:
                out = Node(op="add", inputs=folded_inputs)
                self._copy_meta(node, out)

        elif node.op == "matmul":
            a, b = folded_inputs

            if a.op == "const" and b.op == "const":
                self._log(
                    f"[fold] const @ const -> const  "
                    f"({a.value.shape} @ {b.value.shape})"
                )
                out = self.graph.Const(a.value @ b.value)
            elif folded_inputs == node.inputs:
                out = node
            else:
                out = Node(op="matmul", inputs=folded_inputs)
                self._copy_meta(node, out)

        elif node.op == "relu":
            a = folded_inputs[0]

            if a.op == "const":
                self._log(
                    f"[fold] relu(const) -> const  ({a.value.shape})"
                )
                out = self.graph.Const(np.maximum(a.value, 0))
            elif folded_inputs[0] is node.inputs[0]:
                out = node
            else:
                out = Node(op="relu", inputs=folded_inputs)
                self._copy_meta(node, out)

        elif node.op == "fused_mma_bias_relu":
            x, w, b = folded_inputs
            if x.op == "const" and w.op == "const" and b.op == "const":
                self._log(
                    f"[fold] fused_mma_bias_relu(all-const) -> const  "
                    f"({x.value.shape} @ {w.value.shape} + {b.value.shape})"
                )
                out = self.graph.Const(np.maximum((x.value @ w.value) + b.value, 0))
            elif folded_inputs == node.inputs:
                out = node
            else:
                out = Node(op="fused_mma_bias_relu", inputs=folded_inputs)
                self._copy_meta(node, out)

        elif node.op in ("conv2d", "max_pool2d", "batch_norm",
                         "softmax", "reshape", "transpose"):
            # Element-wise passthrough for ops without const-folding (yet)
            if folded_inputs == node.inputs:
                out = node
            else:
                out = Node(op=node.op, inputs=folded_inputs,
                          attrs=dict(node.attrs))
                self._copy_meta(node, out)

        else:
            raise ValueError(f"Unknown op: {node.op!r}")

        memo[node] = out
        return out

    # ------------------------------------------------------------------
    # Operator fusion
    # ------------------------------------------------------------------

    def fuse(self, node: Node, memo: dict[Node, Node] | None = None) -> Node:
        """Bottom-up operator fusion.

        Merges ``matmul -> add -> relu`` patterns into a single
        ``fused_mma_bias_relu`` node.
        """
        if memo is None:
            memo = {}

        if node in memo:
            return memo[node]

        if node.op in ("input", "const"):
            memo[node] = node
            return node

        fused_inputs = [self.fuse(inp, memo) for inp in node.inputs]

        out: Node

        if node.op == "relu":
            child = fused_inputs[0]
            if child.op == "add":
                left, right = child.inputs
                # Try both orderings: matmul + bias  OR  bias + matmul
                if left.op == "matmul":
                    self._log("[fuse] matmul + add + relu -> fused_mma_bias_relu")
                    out = Node(
                        op="fused_mma_bias_relu",
                        inputs=[left.inputs[0], left.inputs[1], right],
                    )
                    self._copy_meta(node, out)
                    memo[node] = out
                    return out
                if right.op == "matmul":
                    self._log("[fuse] matmul + add + relu -> fused_mma_bias_relu")
                    out = Node(
                        op="fused_mma_bias_relu",
                        inputs=[right.inputs[0], right.inputs[1], left],
                    )
                    self._copy_meta(node, out)
                    memo[node] = out
                    return out

        elif node.op == "add":
            # Conv2D + add + relu fusion: Add(Conv2D(x,w), b) → FusedConvBias
            child_left, child_right = fused_inputs
            if child_left.op == "conv2d" and child_right.op == "const":
                self._log("[fuse] conv2d + add -> fused_conv_bias")
                out = Node(op="fused_conv_bias_relu",
                          inputs=child_left.inputs + [child_right],
                          attrs=dict(child_left.attrs))
                self._copy_meta(node, out)
                memo[node] = out
                return out
            if child_right.op == "conv2d" and child_left.op == "const":
                self._log("[fuse] conv2d + add -> fused_conv_bias")
                out = Node(op="fused_conv_bias_relu",
                          inputs=child_right.inputs + [child_left],
                          attrs=dict(child_right.attrs))
                self._copy_meta(node, out)
                memo[node] = out
                return out

        elif node.op == "relu":
            child = fused_inputs[0]
            # FusedConvBias + relu → fused_conv_bias_relu
            if child.op == "fused_conv_bias":
                self._log("[fuse] fused_conv_bias + relu -> fused_conv_bias_relu")
                out = Node(op="fused_conv_bias_relu",
                          inputs=child.inputs,
                          attrs=dict(child.attrs))
                self._copy_meta(node, out)
                memo[node] = out
                return out
            # BatchNorm + relu fusion
            if child.op == "batch_norm":
                self._log("[fuse] batch_norm + relu -> fused_bn_relu")
                out = Node(op="fused_bn_relu",
                          inputs=child.inputs,
                          attrs=dict(child.attrs))
                self._copy_meta(node, out)
                memo[node] = out
                return out

        if fused_inputs == node.inputs:
            out = node
        else:
            out = Node(op=node.op, inputs=fused_inputs, attrs=dict(node.attrs))
            self._copy_meta(node, out)

        memo[node] = out
        return out

    # ------------------------------------------------------------------
    # Dead code elimination
    # ------------------------------------------------------------------

    def dce(self, output: Node) -> Node:
        """Remove nodes from ``graph.nodes`` that are not reachable from *output*.

        Returns *output* unchanged (the graph topology stays the same; only the
        node list is trimmed).
        """
        reachable = self.graph.reachable_from(output)
        removed = [n for n in self.graph.nodes if n not in reachable]

        if removed:
            names = []
            for n in removed[:8]:
                names.append(n.name or n.op)
            suffix = f" ... (+{len(removed) - 8})" if len(removed) > 8 else ""
            self._log(f"[dce] removed {len(removed)} dead node(s): {', '.join(names)}{suffix}")
            # Trim the graph's node list to reachable-only.
            self.graph.nodes[:] = [n for n in self.graph.nodes if n in reachable]
        else:
            self._log("[dce] no dead nodes found")

        return output

    # ------------------------------------------------------------------
    # Standard pipeline
    # ------------------------------------------------------------------

    def run(self) -> Node:
        """Run the standard optimization pipeline: fold -> fuse -> dce.

        Returns the optimized output node.
        """
        if self.graph.output is None:
            raise ValueError("Graph has no output node set")

        folded = self.fold(self.graph.output)
        fused = self.fuse(folded)
        result = self.dce(fused)
        self.graph.output = result
        return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _label(node: Node) -> str:
    """Short label for log messages."""
    if node.name:
        return f"'{node.name}'"
    if node.op == "const" and isinstance(node.value, np.ndarray):
        return f"const{node.value.shape}"
    return node.op
