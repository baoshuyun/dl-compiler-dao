# mini-dl-compiler: Linalg Dialect
# High-level structured operations on tensors (matmul, add, relu)
# Before tiling and lowering, these ops preserve tensor-level semantics

from ir import Value, Operation, TensorType, FloatType


class LinalgDialect:
    """Builder for linalg-level operations.

    Each static method constructs an Operation with the appropriate
    op_type, inputs, and results. These are the "prima materia" of
    the compiler -- the highest-level IR before progressive lowering.
    """

    @staticmethod
    def matmul(lhs: Value, rhs: Value) -> Operation:
        """Matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j].

        Args:
            lhs: Value of type tensor<M x K>
            rhs: Value of type tensor<K x N>

        Returns:
            Operation producing tensor<M x N>
        """
        M, K = lhs.type.shape
        N = rhs.type.shape[1]
        result = Value("C", TensorType((M, N)))
        op = Operation(
            "linalg.matmul",
            inputs=[lhs, rhs],
            results=[result],
        )
        return op

    @staticmethod
    def add(lhs: Value, rhs: Value) -> Operation:
        """Element-wise addition: C[i] = A[i] + B[i].

        Args:
            lhs: Value of type tensor<...>
            rhs: Value of type tensor<...> (same shape)

        Returns:
            Operation producing tensor of the same shape
        """
        return Operation(
            "linalg.add",
            inputs=[lhs, rhs],
            results=[Value("out", lhs.type)],
        )

    @staticmethod
    def relu(x: Value) -> Operation:
        """ReLU activation: C[i] = max(0, x[i]).

        Args:
            x: Value of type tensor<...>

        Returns:
            Operation producing tensor of the same shape
        """
        return Operation(
            "linalg.relu",
            inputs=[x],
            results=[Value("relu_out", x.type)],
        )


# -- Factory functions for building IR by hand --

def make_value(name: str, shape: tuple) -> Value:
    """Create a Value with a TensorType."""
    return Value(name, TensorType(shape))


def make_float_value(name: str, bits: int = 32) -> Value:
    """Create a Value with a FloatType."""
    return Value(name, FloatType(bits))
