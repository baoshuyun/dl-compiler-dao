"""Unit tests for the IR module."""

import numpy as np
import pytest

from compiler.ir import Graph, Node


class TestNode:
    def test_identity_equality(self):
        """Nodes use identity, not value, for equality."""
        a = Node(op="add")
        b = Node(op="add")
        assert a != b
        assert a is not b
        assert hash(a) != hash(b)

    def test_set_membership(self):
        """Identical-attribute nodes are distinct in sets."""
        a = Node(op="const", value=np.array([1, 2, 3]))
        b = Node(op="const", value=np.array([1, 2, 3]))
        s = {a, b}
        assert len(s) == 2

    def test_repr(self):
        n = Node(op="input", name="x")
        r = repr(n)
        assert "input" in r
        assert "x" in r

    def test_no_name_repr(self):
        n = Node(op="add")
        assert repr(n)  # doesn't crash


class TestGraph:
    def test_add_and_retrieve(self):
        g = Graph()
        n = g.Input("x")
        assert n.op == "input"
        assert n.name == "x"
        assert n in g.nodes

    def test_const_stores_numpy(self):
        g = Graph()
        n = g.Const([1, 2, 3])
        assert isinstance(n.value, np.ndarray)
        assert np.array_equal(n.value, [1, 2, 3])

    def test_add_node(self):
        g = Graph()
        a = g.Input("a")
        b = g.Input("b")
        s = g.Add(a, b)
        assert s.op == "add"
        assert s.inputs == [a, b]

    def test_matmul_node(self):
        g = Graph()
        a = g.Input("a")
        b = g.Input("b")
        m = g.MatMul(a, b)
        assert m.op == "matmul"
        assert m.inputs == [a, b]

    def test_relu_node(self):
        g = Graph()
        x = g.Input("x")
        r = g.ReLU(x)
        assert r.op == "relu"
        assert r.inputs == [x]

    def test_fused_mma_bias_relu(self):
        g = Graph()
        x = g.Input("x")
        w = g.Const([[1, 2]])
        b = g.Const([0])
        f = g.FusedMMA_Bias_ReLU(x, w, b)
        assert f.op == "fused_mma_bias_relu"
        assert f.inputs == [x, w, b]

    def test_input_empty_name_raises(self):
        g = Graph()
        with pytest.raises(ValueError, match="non-empty"):
            g.Input("")

    def test_factory_rejects_non_node(self):
        g = Graph()
        x = g.Input("x")
        with pytest.raises(TypeError):
            g.Add(x, "not_a_node")  # type: ignore[arg-type]

    def test_topological_order_linear(self):
        g = Graph()
        x = g.Input("x")
        w = g.Const([1, 2])
        y = g.Add(x, w)
        order = g.topological_order(y)
        # Inputs before consumers.
        assert order.index(x) < order.index(y)
        assert order.index(w) < order.index(y)

    def test_topological_order_diamond(self):
        g = Graph()
        x = g.Input("x")
        a = g.Add(x, g.Const(1))
        b = g.Add(x, g.Const(2))
        out = g.Add(a, b)
        order = g.topological_order(out)
        assert order.index(a) < order.index(out)
        assert order.index(b) < order.index(out)

    def test_cycle_detection(self):
        g = Graph()
        x = g.Input("x")
        # Manually create a cycle.
        a = Node(op="add", inputs=[x])
        a.inputs.append(a)  # self-loop
        g.nodes.append(a)
        with pytest.raises(RecursionError, match="cycle"):
            g.topological_order(a)

    def test_reachable_from(self):
        g = Graph()
        x = g.Input("x")
        w = g.Const(5)
        dead = g.Const(99)
        out = g.Add(x, w)
        reachable = g.reachable_from(out)
        assert x in reachable
        assert w in reachable
        assert out in reachable
        assert dead not in reachable

    def test_validate_ok(self):
        g = Graph()
        x = g.Input("x")
        g.output = x
        assert g.validate() == []

    def test_validate_warns_orphan_output(self):
        g = Graph()
        x = g.Input("x")
        orphan = Node(op="add")
        g.output = orphan
        warnings = g.validate()
        assert len(warnings) >= 1
        assert "output" in warnings[0].lower()

    def test_output_setter(self):
        g = Graph()
        x = g.Input("x")
        g.output = x
        assert g.output is x
