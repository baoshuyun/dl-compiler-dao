# NPU_Soft_Hard_Stack

> NPU 编译前端 — AST → TaskGraph → CPU/NPU 双 Backend

## Pipeline

```
AST 表达式 → Compiler → TaskGraph (SSA 线性 IR) → Backend
                                                      ├── CPUBackend (Python 解释器)
                                                      └── NPUBackend → ISAMapper → NPU 硬件
```

## 快速开始

```bash
pip install -e "C:\Users\cyy\Desktop\AI_Compiler_Project\mini-dl-compiler"
pip install -e .
python main.py
```

### CPU 示例

```python
from npu_stack import Var, relu, compile_and_run
# relu(w * x + b)
result = compile_and_run(relu(Var("w") * Var("x") + Var("b")),
                         {"w": [2, 0, -1], "x": [1, 2, 3], "b": [1, 1, 1]})
# → [3, 1, 0]
```

### NPU 示例

```python
from npu_stack import Var, matmul, relu, compile_and_run
# relu(X @ W) on NPU
result = compile_and_run(relu(matmul(Var("x"), Var("w"))),
                         {"x": [[1]*4]*4, "w": [[1]*4]*4},
                         backend="npu")
# → [[4,4,4,4], [4,4,4,4], [4,4,4,4], [4,4,4,4]]
```

## 项目结构

```
NPU_Soft_Hard_Stack/
├── npu_stack/
│   ├── ast.py          # Var, Const, Add, Mul, MatMul, Relu
│   ├── ir.py           # Compiler: AST → TaskGraph
│   ├── shape.py        # Shape 推导 + 校验
│   ├── errors.py       # CompilerError, ShapeError, BackendError
│   └── backends/
│       ├── cpu.py      # CPUBackend (向量广播)
│       └── npu.py      # NPUBackend → ISAMapper
├── tests/              # 23 个测试
├── main.py
└── pyproject.toml
```

## 开发

```bash
pytest  # 23 passed
```
