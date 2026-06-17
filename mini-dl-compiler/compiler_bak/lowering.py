# mini-dl-compiler: Progressive Lowering Passes
# Linalg -> Affine -> SCF -> LLVM dialect chain
# Each pass lowers one abstraction level closer to machine code

from ir import Value, Operation, TensorType, MemRefType, FloatType, IntegerType
from typing import List


# -- Simulated dialect operations for each level --

class affine:
    """Affine dialect: structured loops with affine access patterns.

    Key property: loop bounds and array indices are affine functions
    of surrounding loop induction variables. Enables polyhedral analysis.
    """

    @staticmethod
    def for_op(lb: int, ub: int, step: int, name: str) -> Operation:
        """Create an affine.for loop: for i = lb to ub step step."""
        iv = Value(f"iv_{name}", IntegerType(64))
        return Operation(
            "affine.for",
            inputs=[],
            results=[iv],
            attrs={"lower_bound": lb, "upper_bound": ub, "step": step},
            regions=[{"body": {"arguments": [iv], "operations": []}}],
        )

    @staticmethod
    def load(memref: Value, indices: List[Value]) -> Operation:
        """Affine load from memref at affine indices."""
        result = Value("load_val", FloatType(32))
        return Operation(
            "affine.load",
            inputs=[memref] + indices,
            results=[result],
        )

    @staticmethod
    def store(value: Value, memref: Value, indices: List[Value]):
        """Affine store to memref at affine indices."""
        return Operation(
            "affine.store",
            inputs=[value, memref] + indices,
            results=[],
        )


class scf:
    """Structured Control Flow dialect: generic loops and conditionals.

    Less analyzable than affine (no guaranteed affine access patterns),
    but supports arbitrary control flow including while loops and if-else.
    """

    @staticmethod
    def for_op(lb, ub, step, iter_args, name: str) -> Operation:
        """Create an scf.for loop. iter_args carry values across iterations."""
        return Operation(
            "scf.for",
            inputs=iter_args or [],
            results=[],
            attrs={"lower_bound": lb, "upper_bound": ub, "step": step},
            regions=[{"body": {"operations": []}}],
        )


class memref:
    """MemRef dialect: explicit memory allocation and deallocation.

    Introduced during bufferization -- tensor semantics are lowered
    to explicit memory buffers with alloc/dealloc.
    """

    @staticmethod
    def alloc(shape: tuple, element_type=FloatType(32)) -> Operation:
        """Allocate a memref buffer of given shape."""
        result = Value("buf", MemRefType(shape))
        return Operation(
            "memref.alloc",
            inputs=[],
            results=[result],
            attrs={"shape": shape},
        )


class arith:
    """Arithmetic dialect: scalar arithmetic operations independent of memory."""

    @staticmethod
    def addf(lhs: Value, rhs: Value) -> Operation:
        result = Value("addf_res", FloatType(32))
        return Operation("arith.addf", [lhs, rhs], [result])

    @staticmethod
    def mulf(lhs: Value, rhs: Value) -> Operation:
        result = Value("mulf_res", FloatType(32))
        return Operation("arith.mulf", [lhs, rhs], [result])

    @staticmethod
    def maxf(lhs: Value, rhs: Value) -> Operation:
        """Maximum of two floats. Uses 0.0 as RHS for ReLU: max(x, 0.0)."""
        result = Value("maxf_res", FloatType(32))
        return Operation("arith.maxf", [lhs, rhs], [result])



# -- Lowering Passes --

class Pass:
    """Base class for all compiler passes."""
    pass


class LinalgToAffineLowering(Pass):
    """Lower linalg.matmul to 3-level nested affine.for loops.

    linalg.matmul(A[M,K], B[K,N]) -> C[M,N]
    becomes:
        alloc memrefs for A, B, C
        affine.for i=0..M:
            affine.for j=0..N:
                affine.for k=0..K:
                    load A[i,k], B[k,j]
                    mul, add, store to C[i,j]
    """

    def _lower_matmul(self, op: Operation) -> List[Operation]:
        M, K = op.inputs[0].type.shape
        N = op.inputs[1].type.shape[1]

        # Allocate memref buffers (simulated)
        alloc_a = memref.alloc((M, K))
        alloc_b = memref.alloc((K, N))
        alloc_c = memref.alloc((M, N))

        ops: List[Operation] = [alloc_a, alloc_b, alloc_c]

        return ops

    def run(self, module_op: Operation) -> Operation:
        """Walk all ops and lower linalg.matmul to affine loops."""
        # In a real implementation, this would walk the IR tree and
        # replace each linalg.matmul with its affine loop nest.
        return module_op


class AffineToSCFLowering(Pass):
    """Lower affine.for to scf.for.

    affine.for carries structured loop information for polyhedral analysis.
    scf.for is the generic counterpart used after polyhedral transforms.
    The loop body transfers unchanged; only the op wrapper changes.
    """

    def _convert(self, op: Operation) -> Operation:
        lb = op.attributes["lower_bound"]
        ub = op.attributes["upper_bound"]
        step = op.attributes["step"]
        new_for = scf.for_op(lb, ub, step, None, op.attributes.get("name", "loop"))
        # Transfer body: arguments and operations unchanged
        if op.regions:
            new_for.regions[0]["body"]["arguments"] = op.regions[0]["body"]["arguments"]
            new_for.regions[0]["body"]["operations"] = op.regions[0]["body"]["operations"]
        return new_for

    def run(self, module_op: Operation) -> Operation:
        return module_op


class SCFToLLVMLowering(Pass):
    """Lower arith operations to LLVM dialect equivalents.

    arith.addf -> llvm.fadd
    arith.mulf -> llvm.fmul
    arith.maxf -> llvm.maxf

    After this pass, the IR is at a level directly translatable to
    LLVM IR, which can then be JIT-compiled or emitted as object code.
    """

    def run(self, module_op: Operation) -> Operation:
        """Walk all ops and rewrite arith.* to llvm.*."""
        # In production, this uses an IR walker. Here we show the mapping.
        return module_op
