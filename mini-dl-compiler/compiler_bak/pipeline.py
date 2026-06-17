# mini-dl-compiler: Pass Pipeline
# Orchestrates the sequence of compiler passes from high-level IR to target code

from lowering import (
    LinalgToAffineLowering,
    AffineToSCFLowering,
    SCFToLLVMLowering,
    Pass,
)
from transforms import GraphOptimizer, Node
from typing import List


class PassPipeline:
    """Ordered sequence of compiler passes.

    Each pass transforms the IR one step closer to the target.
    Passes are registered in order and executed sequentially.
    """

    def __init__(self):
        self.passes: List[Pass] = []

    def add_pass(self, p: Pass):
        """Append a pass to the pipeline."""
        self.passes.append(p)
        return self

    def run(self, module_op):
        """Execute all passes in order on the module."""
        result = module_op
        for p in self.passes:
            result = p.run(result)
        return result


class LinalgFusionPass(Pass):
    """A pass that runs graph-level fusion before lowering.

    This is conceptually a graph optimization pass, not a dialect
    lowering pass. It operates on Node graphs rather than IR Operations.
    """

    def __init__(self):
        self.optimizer = GraphOptimizer()

    def run(self, nodes: List[Node]) -> List[Node]:
        nodes = self.optimizer.fuse_graph(nodes)
        return nodes


def create_default_pipeline() -> PassPipeline:
    """Factory for the standard lowering pipeline.

    Pipeline stages:
    1. LinalgFusionPass   - graph-level: fuse Add+ReLU, DCE
    2. LinalgToAffineLowering - linalg.matmul -> affine.for loops
    3. AffineToSCFLowering     - affine.for -> scf.for
    4. SCFToLLVMLowering       - arith.* -> llvm.* (ready for codegen)

    After LLVM lowering, the codegen stage (RVVCodegen or CppCodegen)
    emits target-specific source code.
    """
    pipeline = PassPipeline()
    pipeline.add_pass(LinalgFusionPass())
    pipeline.add_pass(LinalgToAffineLowering())
    pipeline.add_pass(AffineToSCFLowering())
    pipeline.add_pass(SCFToLLVMLowering())
    return pipeline
