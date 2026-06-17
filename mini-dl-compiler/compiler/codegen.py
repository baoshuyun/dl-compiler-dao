"""NumPy code generation from the optimized IR."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .ir import Graph, Node
from .optimizer import Optimizer


class NumpyCompiler:
    """Compiles an optimized computation graph into a Python function backed by NumPy."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._memo: dict[Node, str] = {}
        self._tmp_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(
        self,
        output: Node,
        *,
        graph: Graph | None = None,
        optimize: bool = True,
    ) -> tuple[Callable[..., Any], str, Node, list[str]]:
        """Compile *output* (and its dependencies) into a callable.

        Args:
            output: Root node of the computation.
            graph: Optional graph (used for DCE when *optimize* is True).
            optimize: Run the standard optimization pipeline before codegen.

        Returns:
            (callable, source_code, optimized_output_node, optimization_logs)
        """
        logs: list[str] = []

        if optimize and graph is not None:
            opt = Optimizer(graph)
            graph.output = output
            output = opt.run()
            logs = opt.logs

        args = self._collect_inputs(output)
        self._reset()

        self.lines = [
            "import numpy as np",
            "",
            f"def compiled({', '.join(args)}):",
        ]

        out_var = self._emit(output)
        self.lines.append(f"    return {out_var}")

        source = "\n".join(self.lines)

        namespace: dict[str, Any] = {}
        exec(source, namespace)
        return namespace["compiled"], source, output, logs

    def compile_to_source(self, output: Node, *, optimize: bool = True) -> str:
        """Return only the generated source code (no exec)."""
        _, source, _, _ = self.compile(output, optimize=optimize)
        return source

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._memo.clear()
        self._tmp_counter = 0

    def _new_tmp(self) -> str:
        self._tmp_counter += 1
        return f"v{self._tmp_counter}"

    @staticmethod
    def _encode(value: np.ndarray[Any, Any]) -> str:
        return f"np.array({value.tolist()!r}, dtype=np.{value.dtype.name})"

    def _collect_inputs(self, root: Node) -> list[str]:
        ordered: list[str] = []
        seen_nodes: set[Node] = set()
        seen_names: set[str] = set()

        def visit(n: Node) -> None:
            if n in seen_nodes:
                return
            seen_nodes.add(n)
            if n.op == "input" and n.name not in seen_names:
                ordered.append(n.name)
                seen_names.add(n.name)
            for child in n.inputs:
                visit(child)

        visit(root)
        return ordered

    def _emit(self, node: Node) -> str:
        """Emit code for *node* and return the variable name holding its value."""
        if node in self._memo:
            return self._memo[node]

        if node.op == "input":
            self._memo[node] = node.name
            return node.name

        if node.op == "const":
            var = self._new_tmp()
            self.lines.append(f"    {var} = {self._encode(node.value)}")
            self._memo[node] = var
            return var

        # Gather operand variable names
        operands = [self._emit(inp) for inp in node.inputs]

        var = self._new_tmp()

        if node.op == "add":
            self.lines.append(f"    {var} = {operands[0]} + {operands[1]}")
        elif node.op == "matmul":
            self.lines.append(f"    {var} = {operands[0]} @ {operands[1]}")
        elif node.op == "relu":
            self.lines.append(f"    {var} = np.maximum({operands[0]}, 0)")
        elif node.op == "fused_mma_bias_relu":
            x, w, b = operands
            self.lines.append(f"    {var} = np.maximum(({x} @ {w}) + {b}, 0)")
        elif node.op == "conv2d":
            x, w = operands
            strides = node.attrs.get("strides", (1, 1))
            padding = node.attrs.get("padding", (0, 0))
            self.lines.extend(self._emit_conv2d(var, x, w, strides, padding))
        elif node.op == "max_pool2d":
            ks = node.attrs.get("kernel_size", (2, 2))
            st = node.attrs.get("strides", ks)
            self.lines.extend(self._emit_maxpool2d(var, operands[0], ks, st))
        elif node.op == "batch_norm":
            x, s, b, m, v = operands
            eps = node.attrs.get("epsilon", 1e-5)
            self.lines.append(
                f"    {var} = ({x} - {m}) / np.sqrt({v} + {eps}) * {s} + {b}")
        elif node.op == "fused_bn_relu":
            x, s, b, m, v = operands
            eps = node.attrs.get("epsilon", 1e-5)
            self.lines.append(
                f"    {var} = np.maximum(({x} - {m}) / np.sqrt({v} + {eps}) * {s} + {b}, 0)")
        elif node.op == "softmax":
            axis = node.attrs.get("axis", -1)
            self.lines.append(
                f"    {var} = np.exp({operands[0]} - {operands[0]}.max(axis={axis}, keepdims=True))")
            self.lines.append(
                f"    {var} = {var} / {var}.sum(axis={axis}, keepdims=True)")
        elif node.op == "reshape":
            shape = node.attrs["shape"]
            self.lines.append(f"    {var} = {operands[0]}.reshape({shape})")
        elif node.op == "transpose":
            axes = node.attrs.get("axes", None)
            axes_str = str(axes) if axes else ""
            self.lines.append(f"    {var} = {operands[0]}.transpose({axes_str})" if axes else
                            f"    {var} = {operands[0]}.T")
        elif node.op == "fused_conv_bias_relu":
            x, w, b = operands[:3]
            strides = node.attrs.get("strides", (1, 1))
            padding = node.attrs.get("padding", (0, 0))
            tmp = self._new_tmp()
            self.lines.extend(self._emit_conv2d(tmp, x, w, strides, padding))
            self.lines.append(f"    {var} = np.maximum({tmp} + {b}, 0)")
        else:
            raise ValueError(f"Unknown op: {node.op!r}")

        self._memo[node] = var
        return var

    # ── Op-specific emitters ─────────────────────────────────────

    @staticmethod
    def _emit_conv2d(var: str, x: str, w: str,
                     strides: tuple[int, int],
                     padding: tuple[int, int]) -> list[str]:
        """Emit im2col-based conv2d NumPy code.

        Returns lines to append (caller adds to self.lines).
        """
        # Simple implementation using nested loops for clarity.
        # In production, use np.lib.stride_tricks.as_strided + tensordot.
        ph, pw = padding
        sh, sw = strides
        lines = [
            f"    # conv2d: {x} @ {w} strides={strides} pad={padding}",
            f"    _x = {x}",
            f"    _w = {w}",
        ]
        if ph > 0 or pw > 0:
            lines.append(
                f"    _x = np.pad(_x, ((0,0),({ph},{ph}),({pw},{pw}),(0,0)))")
        lines.extend([
            f"    _n, _h, _w_in, _c_in = _x.shape",
            f"    _kh, _kw, _c_in2, _c_out = _w.shape",
            f"    _oh = (_h - _kh) // {sh} + 1",
            f"    _ow = (_w_in - _kw) // {sw} + 1",
            f"    # im2col + matmul",
            f"    _cols = np.zeros((_n * _oh * _ow, _kh * _kw * _c_in), dtype=_x.dtype)",
            f"    _idx = 0",
            f"    for _i in range(_n):",
            f"        for _r in range(0, _oh):",
            f"            for _c in range(0, _ow):",
            f"                _patch = _x[_i, _r*{sh}:_r*{sh}+_kh, _c*{sw}:_c*{sw}+_kw, :]",
            f"                _cols[_idx] = _patch.ravel()",
            f"                _idx += 1",
            f"    _w_flat = _w.reshape(-1, _c_out)",
            f"    {var} = (_cols @ _w_flat).reshape(_n, _oh, _ow, _c_out)",
        ])
        return lines

    @staticmethod
    def _emit_maxpool2d(var: str, x: str,
                        kernel_size: tuple[int, int],
                        strides: tuple[int, int]) -> list[str]:
        """Emit max_pool2d NumPy code."""
        kh, kw = kernel_size
        sh, sw = strides
        return [
            f"    # max_pool2d: kernel={kernel_size} strides={strides}",
            f"    _x = {x}",
            f"    _n, _h, _w, _c = _x.shape",
            f"    _oh = (_h - {kh}) // {sh} + 1",
            f"    _ow = (_w - {kw}) // {sw} + 1",
            f"    {var} = np.zeros((_n, _oh, _ow, _c), dtype=_x.dtype)",
            f"    for _r in range(_oh):",
            f"        for _c in range(_ow):",
            f"            {var}[:, _r, _c, :] = _x[:, _r*{sh}:_r*{sh}+{kh}, _c*{sw}:_c*{sw}+{kw}, :].max(axis=(1,2))",
        ]
