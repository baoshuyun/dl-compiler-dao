"""Typed errors for the NPU compiler stack."""


class CompilerError(Exception):
    """Raised when compilation fails (unknown AST node, etc.)."""
    pass


class ShapeError(Exception):
    """Raised when shape inference detects a mismatch."""
    pass


class BackendError(Exception):
    """Raised when a backend cannot execute the given graph."""
    pass
