"""Backend implementations for the NPU compiler stack."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..ir import TaskGraph


class Backend(ABC):
    """Abstract backend interface."""

    @abstractmethod
    def run(self, graph: TaskGraph, inputs: dict | None = None) -> Any:
        """Execute a compiled TaskGraph and return the result."""
        ...


from .cpu import CPUBackend  # noqa: E402, F401

__all__ = ["Backend", "CPUBackend"]
