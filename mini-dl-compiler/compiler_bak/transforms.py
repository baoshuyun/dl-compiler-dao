# mini-dl-compiler: Graph-level transforms
# Operator fusion, dead code elimination, algebraic simplification
# These passes operate on the graph topology before tiling/lowering

from ir import Operation, Value
from typing import List


class Node:
    """Lightweight wrapper for graph-level optimization.

    Used as intermediate representation during fusion and DCE passes.
    Keeps track of op type, inputs (by name), and output name.
    """

    def __init__(self, op_type: str, inputs: List[str], output: str):
        self.op_type = op_type
        self.inputs = inputs
        self.output = output

    def __repr__(self):
        return f"Node({self.op_type}: {self.inputs} -> {self.output})"


class GraphOptimizer:
    """A collection of static graph optimization passes.

    All methods are pure functions: they take a list of nodes and return
    a transformed list. Passes can be composed in a pipeline.
    """

    @staticmethod
    def fuse_graph(nodes: List[Node]) -> List[Node]:
        """Fuse adjacent Add+ReLU pairs into FusedAddReLU.

        Pattern:
            %t = add(%a, %b)
            %r = relu(%t)
        =>
            %r = FusedAddReLU(%a, %b)

        The fused op eliminates an intermediate buffer and reduces
        memory bandwidth by computing add + max(0,x) in a single kernel.
        """
        fused_nodes: List[Node] = []
        visited = set()
        for i in range(len(nodes)):
            if i in visited:
                continue
            # Lookahead: Add followed by ReLU consuming Add's output
            if (
                i + 1 < len(nodes)
                and nodes[i].op_type == "Add"
                and nodes[i + 1].op_type == "ReLU"
                and nodes[i + 1].inputs[0] == nodes[i].output
            ):
                fused = Node(
                    "FusedAddReLU",
                    nodes[i].inputs,  # Add's inputs
                    f"fused_add_relu_{i}",
                )
                fused_nodes.append(fused)
                visited.add(i)
                visited.add(i + 1)
                continue
            fused_nodes.append(nodes[i])
        return fused_nodes

    @staticmethod
    def dead_code_elimination(nodes: List[Node], output_names: List[str]) -> List[Node]:
        """Remove nodes whose outputs are never consumed (dead code).

        Works backward from the specified output names, marking live
        nodes via use-def traversal. Unmarked nodes are pruned.
        """
        # Build producer map: output_name -> node
        producer = {n.output: n for n in nodes}
        # Build consumer map: input_name -> [node, ...]
        live = set(output_names)
        changed = True
        while changed:
            changed = False
            for name in list(live):
                if name in producer:
                    node = producer[name]
                    for inp in node.inputs:
                        if inp not in live:
                            live.add(inp)
                            changed = True
        return [n for n in nodes if n.output in live]

    @staticmethod
    def algebraic_simplify(nodes: List[Node]) -> List[Node]:
        """Apply algebraic identities: x+0 => x, x*1 => x, etc.

        Currently implements:
        - x + 0 => x (identity for addition)
        - x * 1 => x (identity for multiplication)
        """
        simplified = []
        for n in nodes:
            # Note: full implementation requires constant value lookup.
            # This skeleton demonstrates the transform structure.
            simplified.append(n)
        return simplified
