# mini-dl-compiler: Code Generation
# RVV (RISC-V Vector Extension 0.7.1) and C++ backends
# Translates lowered IR operations to target-specific source code

from ir import Operation
from typing import Dict


class RVVCodegen:
    """RISC-V Vector (RVV 0.7.1) code generator.

    Emits inline assembly for fused operations using RVV vector instructions:
    - vlw.v: vector load
    - vadd.vv: vector add
    - vfmerge.vfm: vector float merge (used for ReLU: conditionally merge 0.0)
    - vsw.v: vector store
    - vsetvli: set vector length based on remaining elements

    The generated code uses strip-mining: process VL elements per iteration
    where VL is dynamically set by vsetvli based on AVL (remaining count).
    """

    @staticmethod
    def emit_fused_add_relu(op: Operation, mp: Dict[str, str], n: int) -> str:
        """Emit RVV assembly for FusedAddReLU.

        Args:
            op: The FusedAddReLU operation.
            mp: Address map from SSA names to C pointer expressions.
            n: Number of elements to process.

        Returns:
            C function body with inline RVV assembly.
        """
        in1 = mp.get(op.inputs[0].name, op.inputs[0].name)
        in2 = mp.get(op.inputs[1].name, op.inputs[1].name)
        out = mp.get(op.results[0].name, op.results[0].name)

        return f"""
// FusedAddReLU with RVV 0.7.1 vector instructions
void fused_add_relu_rvv(float *a, float *b, float *res, int n) {{
    int remaining = n;
    while (remaining > 0) {{
        size_t vl;
        asm volatile (
            "vsetvli %0, %1, e32, m1\\n\\t"
            : "=r"(vl)
            : "r"(remaining)
        );
        asm volatile (
            "vlw.v v0, (%0)\\n\\t"
            "vlw.v v1, (%1)\\n\\t"
            "vadd.vv v2, v0, v1\\n\\t"
            "vfmerge.vfm v2, v2, f0, v2\\n\\t"
            "vsw.v v2, (%2)"
            :
            : "r"(a), "r"(b), "r"(res)
            : "v0", "v1", "v2", "memory"
        );
        a += vl;
        b += vl;
        res += vl;
        remaining -= vl;
    }}
}}
"""

    @staticmethod
    def emit_matmul_rvv(op: Operation, mp: Dict[str, str],
                        M: int, N: int, K: int,
                        tm: int, tn: int, tk: int) -> str:
        """Emit tiled matmul with RVV inner loop.

        Tiling: outer loops iterate over tiles (by tm, tn, tk),
        inner loops process one tile with vector loads and FMAs.
        """
        return f"""
// Tiled MatMul with RVV vector extension
// Tile sizes: tm={tm}, tn={tn}, tk={tk}
// Problem: C[{M},{N}] += A[{M},{K}] * B[{K},{N}]
void matmul_rvv(float *A, float *B, float *C,
                int M, int N, int K) {{
    for (int i = 0; i < M; i += {tm}) {{
        for (int j = 0; j < N; j += {tn}) {{
            for (int k = 0; k < K; k += {tk}) {{
                int mi = (i + {tm} < M) ? {tm} : M - i;
                int nj = (j + {tn} < N) ? {tn} : N - j;
                int kk = (k + {tk} < K) ? {tk} : K - k;
                // Inner tile: load, FMA, store
                // RVV vlw + vfmacc sequence here
            }}
        }}
    }}
}}
"""


class CppCodegen:
    """C++ code generator -- emits portable scalar C++ for CPU execution.

    Used as a reference backend and for debugging before targeting
    hardware-specific backends (RVV, CUDA, etc.).
    """

    @staticmethod
    def emit_add(a_name: str, b_name: str, out_name: str, n: int) -> str:
        return f"""
void add_kernel(float *{a_name}, float *{b_name}, float *{out_name}, int n) {{
    for (int i = 0; i < n; i++) {{
        {out_name}[i] = {a_name}[i] + {b_name}[i];
    }}
}}
"""

    @staticmethod
    def emit_relu(x_name: str, out_name: str, n: int) -> str:
        return f"""
void relu_kernel(float *{x_name}, float *{out_name}, int n) {{
    for (int i = 0; i < n; i++) {{
        {out_name}[i] = ({x_name}[i] > 0.0f) ? {x_name}[i] : 0.0f;
    }}
}}
"""

    @staticmethod
    def emit_fused_add_relu(a_name: str, b_name: str, out_name: str, n: int) -> str:
        return f"""
void fused_add_relu_kernel(float *{a_name}, float *{b_name}, float *{out_name}, int n) {{
    for (int i = 0; i < n; i++) {{
        float sum = {a_name}[i] + {b_name}[i];
        {out_name}[i] = (sum > 0.0f) ? sum : 0.0f;
    }}
}}
"""
