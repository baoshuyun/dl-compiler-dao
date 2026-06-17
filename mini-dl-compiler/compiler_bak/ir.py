# mini-dl-compiler: SSA-based Intermediate Representation
# Core IR data structures: Value, Operation, and use-def chain management
# Part of AI_Compiler_Project

from __future__ import annotations
from typing import Any, List, Optional, Dict


class Value:
    """An SSA value produced by an Operation.

    Each Value has:
    - name: human-readable identifier (e.g. "%c", "%add_0")
    - type: type descriptor (TensorType, FloatType, etc.)
    - defining_op: the Operation that produces this value (None for block arguments)
    - uses: list of Operations that consume this value as input
    """

    def __init__(self, name: str, type_):
        self.name = name
        self.type = type_
        self.defining_op: Optional[Operation] = None
        self.uses: List[Operation] = []

    def __repr__(self):
        return f"%{self.name}:{self.type}"

    def add_use(self, op: Operation):
        """Register that `op` consumes this value as an input."""
        self.uses.append(op)

    def remove_use(self, op: Operation):
        """Remove `op` from the use list."""
        if op in self.uses:
            self.uses.remove(op)


class Operation:
    """The core unit of computation in the IR.

    Each Operation has:
    - op_type: fully qualified operation name (e.g. "linalg.matmul")
    - dialect: extracted from op_type prefix
    - inputs: list of Value(s) consumed
    - results: list of Value(s) produced
    - attributes: key-value metadata (e.g. tile sizes, strides)
    - regions: nested region bodies (for control flow ops)
    """

    _id_counter: int = 0

    def __init__(
        self,
        op_type: str,
        inputs: List[Value],
        results: List[Value],
        attrs: Optional[Dict[str, Any]] = None,
        regions: Optional[List] = None,
    ):
        Operation._id_counter += 1
        self.id = Operation._id_counter
        self.op_type = op_type
        self.dialect = op_type.split(".")[0]
        self.inputs = list(inputs)
        self.results = list(results)
        self.attributes = attrs or {}
        self.regions = regions or []

        # Wire up SSA use-def chains
        for inp in self.inputs:
            inp.add_use(self)
        for r in self.results:
            r.defining_op = self

    def replace_all_uses_with(self, new_results: List[Value]):
        """Replace all uses of this op's results with `new_results`.

        This is the fundamental mechanism for graph rewriting:
        after fusion or DCE, consumers are redirected to new producers.

        Maps old results to new results positionally. Each consumer's
        input list is updated in-place to point to the replacement values.
        """
        for old_val, new_val in zip(self.results, new_results):
            for use_op in old_val.uses[:]:  # iterate over copy
                for i, inp in enumerate(use_op.inputs):
                    if inp is old_val:
                        use_op.inputs[i] = new_val
                        old_val.remove_use(use_op)
                        new_val.add_use(use_op)

    def __repr__(self):
        in_names = [v.name for v in self.inputs]
        out_names = [v.name for v in self.results]
        return f"{self.op_type}({', '.join(in_names)}) -> ({', '.join(out_names)})"


# -- Type System --

class TensorType:
    """Describes a multi-dimensional array type: shape + element type."""
    def __init__(self, shape: tuple, element_type: type = float):
        self.shape = shape
        self.element_type = element_type

    def __repr__(self):
        dims = "x".join(str(d) for d in self.shape)
        return f"tensor<{dims}x{self.element_type.__name__}>"


class FloatType:
    """Scalar floating-point type with specified bit width."""
    def __init__(self, bits: int = 32):
        self.bits = bits

    def __repr__(self):
        return f"f{self.bits}"


class MemRefType:
    """Describes a memory buffer: shape + element type, used after bufferization."""
    def __init__(self, shape: tuple, element_type: type = float):
        self.shape = shape
        self.element_type = element_type

    def __repr__(self):
        dims = "x".join(str(d) for d in self.shape)
        return f"memref<{dims}x{self.element_type.__name__}>"


class IntegerType:
    """Scalar integer type with specified bit width and signedness."""
    def __init__(self, bits: int = 32, signed: bool = True):
        self.bits = bits
        self.signed = signed

    def __repr__(self):
        prefix = "i" if self.signed else "ui"
        return f"{prefix}{self.bits}"
