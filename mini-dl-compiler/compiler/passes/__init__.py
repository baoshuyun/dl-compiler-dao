"""Optimization and lowering passes for the DL compiler.

Each Pass transforms the IR in a specific, verifiable way.
PassManager orchestrates execution with dependency tracking
and optional verification between passes.

Passes:
    TilingPass        — decompose large matmuls with cost model
    MemoryPlanningPass — liveness-aware buffer reuse
    GraphToSSAPass    — lower Graph IR to SSA IR
    QuantizePass      — FP32 → INT8 quantization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ════════════════════════════════════════════════════════════════
# Pass infrastructure
# ════════════════════════════════════════════════════════════════

class Pass(ABC):
    """Abstract compiler pass."""

    name: str = ""

    @abstractmethod
    def run(self, ir: Any, **kwargs: Any) -> Any:
        """Transform the IR. Returns the transformed IR."""
        ...

    def verify(self, before: Any, after: Any) -> list[str]:
        """Optional verification after the pass. Returns list of warnings."""
        return []


@dataclass
class PassResult:
    """Result of running a pass."""
    pass_name: str
    ir: Any
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class PassManager:
    """Orchestrates ordered execution of compiler passes.

    Usage::

        pm = PassManager()
        pm.add(TilingPass(target_rows=4, target_cols=4))
        pm.add(MemoryPlanningPass())
        result = pm.run(ir)
        print(pm.summary())
    """

    def __init__(self) -> None:
        self._passes: list[Pass] = []
        self.results: list[PassResult] = []

    def add(self, p: Pass) -> None:
        self._passes.append(p)

    def run(self, ir: Any) -> Any:
        """Execute all passes in order. Returns the final IR."""
        self.results = []
        current = ir

        for p in self._passes:
            before = current
            current = p.run(current)
            warnings = p.verify(before, current)
            self.results.append(PassResult(
                pass_name=p.name,
                ir=current,
                warnings=warnings,
            ))

        return current

    def summary(self) -> str:
        """Return a human-readable summary of pass execution."""
        lines = ["Pass Pipeline Summary:", "-" * 40]
        for r in self.results:
            status = "✓" if not r.warnings else f"⚠ ({len(r.warnings)} warnings)"
            lines.append(f"  {r.pass_name:<30} {status}")
            for w in r.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


from .tiling import TilingPass  # noqa: E402, F401
from .memory_planning import MemoryPlanningPass  # noqa: E402, F401
from .lowering import GraphToSSAPass  # noqa: E402, F401

__all__ = [
    "Pass", "PassManager", "PassResult",
    "TilingPass", "MemoryPlanningPass", "GraphToSSAPass",
]
