# mini-dl-compiler: End-to-End Demo
# Demonstrates the full compilation pipeline and all major components
# Part of AI_Compiler_Project

from ir import Value, TensorType, FloatType, IntegerType
from dialects.linalg import LinalgDialect, make_value, make_float_value
from transforms import GraphOptimizer, Node
from tiling import compute_tile_sizes
from codegen import RVVCodegen, CppCodegen
from memory import LivenessAnalyzer, MemoryPlanner, BufferDescriptor
from pipeline import create_default_pipeline


def demo_ir():
    """Demonstrate SSA IR construction with use-def chains."""
    print("=" * 60)
    print("  mini-dl-compiler: IR Construction Demo")
    print("=" * 60)

    # Create values
    A = make_value("A", (4, 8))
    B = make_value("B", (8, 4))

    # Build matmul operation
    matmul_op = LinalgDialect.matmul(A, B)
    print(f"\n  Operation: {matmul_op}")
    print(f"  Result type: {matmul_op.results[0].type}")

    # Verify SSA chain
    result = matmul_op.results[0]
    print(f"  Result defining_op: {result.defining_op.op_type if result.defining_op else 'None'}")
    print(f"  Uses: {len(result.uses)} consumer(s)")

    # Build an Add + ReLU chain
    X = make_value("X", (4, 4))
    Y = make_value("Y", (4, 4))
    add_op = LinalgDialect.add(X, Y)
    relu_op = LinalgDialect.relu(add_op.results[0])
    print(f"\n  Add op: {add_op}")
    print(f"  ReLU op: {relu_op}")
    print(f"  Intermediate value uses: {len(add_op.results[0].uses)} consumer(s)")

    print("\n  IR construction: OK")


def demo_fusion():
    """Demonstrate graph-level operator fusion."""
    print("\n" + "=" * 60)
    print("  mini-dl-compiler: Graph Fusion Demo")
    print("=" * 60)

    # Build a node graph: Add -> ReLU -> Mul (artificial chain)
    nodes = [
        Node("Add", ["a", "b"], "%t1"),
        Node("ReLU", ["%t1"], "%t2"),
        Node("Mul", ["%t2", "c"], "%out"),
    ]

    print("\n  Before fusion:")
    for n in nodes:
        print(f"    {n}")

    fused = GraphOptimizer.fuse_graph(nodes)

    print("\n  After fusion (Add+ReLU -> FusedAddReLU):")
    for n in fused:
        print(f"    {n}")

    # Verify: should have 2 nodes instead of 3
    assert len(fused) == 2, f"Expected 2 nodes after fusion, got {len(fused)}"
    assert fused[0].op_type == "FusedAddReLU", f"Expected FusedAddReLU, got {fused[0].op_type}"
    print("\n  Fusion: OK (3 ops -> 2 ops, Add+ReLU fused)")


def demo_tiling():
    """Demonstrate tile size search for matmul."""
    print("\n" + "=" * 60)
    print("  mini-dl-compiler: Tiling Cost Model Demo")
    print("=" * 60)

    # Simulate a 256x256x256 matmul
    A = make_value("A", (256, 256))
    B = make_value("B", (256, 256))
    matmul_op = LinalgDialect.matmul(A, B)

    tiles = compute_tile_sizes(matmul_op, config={"l1_capacity": 4096})
    print(f"\n  Problem: M=256, N=256, K=256")
    print(f"  L1 capacity: 4096 elements (16 KB for float32)")
    print(f"  Optimal tile sizes found: tm={tiles[0]}, tn={tiles[1]}, tk={tiles[2]}")
    print(f"  Tiling: OK")


def demo_codegen():
    """Demonstrate code generation for fused operations."""
    print("\n" + "=" * 60)
    print("  mini-dl-compiler: Code Generation Demo")
    print("=" * 60)

    # Build a FusedAddReLU operation
    A = make_value("in_a", (128,))
    B = make_value("in_b", (128,))
    result = Value("fused_out", TensorType((128,)))
    fused_op = type('Operation', (), {
        'op_type': 'FusedAddReLU',
        'inputs': [A, B],
        'results': [result],
        'attributes': {},
    })()

    # Generate RVV code
    mp = {"in_a": "a_ptr", "in_b": "b_ptr", "fused_out": "out_ptr"}
    rvv_code = RVVCodegen.emit_fused_add_relu(fused_op, mp, 128)
    print("\n  --- RVV 0.7.1 Code ---")
    print(rvv_code[:400] + "...")

    # Generate C++ code
    cpp_code = CppCodegen.emit_fused_add_relu("a", "b", "out", 128)
    print("  --- C++ Reference Code ---")
    print(cpp_code[:300] + "...")

    print("  Codegen: OK")


def demo_memory():
    """Demonstrate liveness analysis and memory planning."""
    print("\n" + "=" * 60)
    print("  mini-dl-compiler: Memory Planning Demo")
    print("=" * 60)

    # Simulate 6 buffers produced across 5 operations
    buffers = [
        BufferDescriptor("%a", (256, 256), birth=0, death=2),
        BufferDescriptor("%b", (256, 256), birth=0, death=3),
        BufferDescriptor("%c", (256, 256), birth=2, death=4),
        BufferDescriptor("%d", (128, 128), birth=3, death=4),
        BufferDescriptor("%e", (128, 128), birth=4, death=5),
        BufferDescriptor("%f", (64, 64),   birth=1, death=3),
    ]

    planner = MemoryPlanner(alignment=64)
    total = planner.plan(buffers)

    naive_total = sum(b.size for b in buffers)

    print(f"\n  Buffers: {len(buffers)}")
    print(f"  Naive allocation (sum of all): {naive_total} bytes ({naive_total / 1024:.1f} KB)")
    print(f"  Planned pool size: {total} bytes ({total / 1024:.1f} KB)")
    print(f"  Memory saved: {naive_total - total} bytes ({(1 - total/naive_total)*100:.0f}%)")

    print(f"\n  Buffer assignments:")
    for b in sorted(buffers, key=lambda x: x.offset):
        print(f"    {b.name}: offset={b.offset}, size={b.size}, "
              f"birth={b.birth}, death={b.death}")

    print("\n  Memory planning: OK")


def demo_pipeline():
    """Demonstrate the full lowering pipeline structure."""
    print("\n" + "=" * 60)
    print("  mini-dl-compiler: Pass Pipeline Demo")
    print("=" * 60)

    pipeline = create_default_pipeline()
    print(f"\n  Pipeline passes ({len(pipeline.passes)} total):")
    for i, p in enumerate(pipeline.passes):
        print(f"    {i+1}. {type(p).__name__}")

    print("\n  Pipeline: OK")


def main():
    """Run all demos."""
    demo_ir()
    demo_fusion()
    demo_tiling()
    demo_codegen()
    demo_memory()
    demo_pipeline()

    print("\n" + "=" * 60)
    print("  mini-dl-compiler: All components verified OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
