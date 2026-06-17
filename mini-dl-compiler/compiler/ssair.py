"""SSA IR for the DL compiler.

Provides fine-grained SSA (Static Single Assignment) IR types for
the lowering phase.  Ported from the pedagogical skeleton and
upgraded with type safety and validation.

Classes:
    Type hierarchy: IRType → TensorType, MemRefType, FloatType, IntegerType
    Value: SSA value with use-def chains
    Operation: SSA operation with dialect tagging, regions support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


# ════════════════════════════════════════════════════════════════
# Types
# ════════════════════════════════════════════════════════════════

class IRType:
    """Base class for IR types."""

    def __repr__(self) -> str:
        return f"<{self}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IRType):
            return NotImplemented
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))


@dataclass(frozen=True)
class TensorType(IRType):
    """Multi-dimensional array type.

    Example: ``TensorType((256, 256), "float32")``
    """
    shape: tuple[int, ...]
    element_type: str = "float32"

    def __str__(self) -> str:
        dims = "×".join(str(d) for d in self.shape)
        return f"tensor<{dims}×{self.element_type}>"

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class MemRefType(IRType):
    """Explicit memory buffer type (post-bufferization).

    Example: ``MemRefType((256, 256), "float32")``
    """
    shape: tuple[int, ...]
    element_type: str = "float32"

    def __str__(self) -> str:
        dims = "×".join(str(d) for d in self.shape)
        return f"memref<{dims}×{self.element_type}>"

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class FloatType(IRType):
    """Scalar floating-point type."""
    bits: int = 32

    def __str__(self) -> str:
        return f"f{self.bits}"


@dataclass(frozen=True)
class IntegerType(IRType):
    """Scalar integer type."""
    bits: int = 32
    signed: bool = True

    def __str__(self) -> str:
        prefix = "i" if self.signed else "ui"
        return f"{prefix}{self.bits}"


# ════════════════════════════════════════════════════════════════
# SSA Value
# ════════════════════════════════════════════════════════════════

@dataclass
class SSAValue:
    """An SSA value in the IR.

    Each Value is defined by exactly one Operation and may be used
    by many Operations.  Use-def chains are maintained automatically
    when Operations are created.
    """
    name: str
    type: IRType = field(default_factory=lambda: FloatType(32))
    defining_op: Operation | None = field(default=None, repr=False)
    uses: list[Operation] = field(default_factory=list, repr=False)

    def add_use(self, op: Operation) -> None:
        if op not in self.uses:
            self.uses.append(op)

    def remove_use(self, op: Operation) -> None:
        if op in self.uses:
            self.uses.remove(op)

    def replace_all_uses_with(self, new_value: SSAValue) -> None:
        """Redirect all users of this value to *new_value*."""
        for op in list(self.uses):
            op.replace_input(self, new_value)
        self.uses.clear()

    def __repr__(self) -> str:
        return f"SSAValue({self.name!r}, {self.type})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


# ════════════════════════════════════════════════════════════════
# Operation
# ════════════════════════════════════════════════════════════════

@dataclass
class Operation:
    """An SSA operation in the IR.

    Operations belong to a dialect (e.g. ``linalg.matmul``) and
    produce zero or more result Values.

    Attributes:
        op_type: Fully-qualified operation name (e.g. ``linalg.matmul``).
        inputs: Input SSA Values.
        results: Output SSA Values (auto-created if not provided).
        attributes: Arbitrary key-value metadata (tile sizes, strides, etc.).
        regions: Nested regions for control flow ops.
        dialect: Auto-extracted from *op_type* prefix.
    """

    op_type: str
    inputs: list[SSAValue] = field(default_factory=list)
    results: list[SSAValue] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    regions: list[list[Operation]] = field(default_factory=list)
    dialect: str = field(init=False)
    _result_counter: ClassVar[int] = 0

    def __post_init__(self) -> None:
        self.dialect = self.op_type.split(".")[0] if "." in self.op_type else ""

        # Auto-create results if none provided
        if not self.results:
            self.results = [self._make_result() for _ in range(self._result_arity())]

        # Wire use-def chains
        for inp in self.inputs:
            inp.add_use(self)
        for r in self.results:
            r.defining_op = self

    def _make_result(self) -> SSAValue:
        Operation._result_counter += 1
        return SSAValue(name=f"%r{Operation._result_counter}")

    def _result_arity(self) -> int:
        """Number of results for this op type (can be overridden per dialect)."""
        return 1

    def replace_input(self, old_val: SSAValue, new_val: SSAValue) -> None:
        """Replace *old_val* with *new_val* in this op's input list."""
        if old_val in self.inputs:
            idx = self.inputs.index(old_val)
            self.inputs[idx] = new_val
            old_val.remove_use(self)
            new_val.add_use(self)

    def replace_all_uses_with(self, new_results: list[SSAValue]) -> None:
        """Replace all uses of this op's results with *new_results*."""
        for old_r, new_r in zip(self.results, new_results):
            old_r.replace_all_uses_with(new_r)

    def __repr__(self) -> str:
        in_names = [v.name for v in self.inputs]
        out_names = [v.name for v in self.results]
        return f"Operation({self.op_type!r}, {in_names} → {out_names})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


# ════════════════════════════════════════════════════════════════
# Utility
# ════════════════════════════════════════════════════════════════

def make_tensor_value(name: str, shape: tuple[int, ...],
                      dtype: str = "float32") -> SSAValue:
    """Create a tensor-typed SSA value."""
    return SSAValue(name=name, type=TensorType(shape, dtype))


def make_memref_value(name: str, shape: tuple[int, ...],
                      dtype: str = "float32") -> SSAValue:
    """Create a memref-typed SSA value."""
    return SSAValue(name=name, type=MemRefType(shape, dtype))
