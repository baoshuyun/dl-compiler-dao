# mini-dl-compiler

> NPU AI 编译优化引擎 — MLIR 风格多级 IR 渐进式下降编译器

## 优化管线

```
Graph IR → ConstantFolding → OperatorFusion → DCE
         → Tiling(束搜索) → MemoryPlanning(Liveness)
         → GraphToSSA → Codegen → NumPy / C++ / NPU ISA
```

## 快速开始

```bash
pip install -e .
python main.py
python main.py --benchmark
```

## 项目结构

```
mini-dl-compiler/
├── compiler/
│   ├── ir.py              # Graph IR: Node, Graph, 14种算子
│   ├── optimizer.py        # Fold + Fuse + DCE
│   ├── codegen.py          # NumPy 代码生成 (conv2d/pool/softmax)
│   ├── viz.py              # DOT/PNG 图可视化
│   ├── ssair.py            # SSA IR: Value, Operation, Type 系统
│   ├── passes/
│   │   ├── tiling.py       # 分块代价模型 + 束搜索
│   │   ├── memory_planning.py  # Liveness + 贪心 Buffer 复用
│   │   └── lowering.py     # Graph IR → SSA IR
│   ├── isa_mapper/         # NPU 指令翻译桥
│   │   ├── isa.py          # ISA 编码 (与 isa_defines.vh 同步)
│   │   ├── scheduler.py    # 依赖分析 + BARRIER 插入
│   │   ├── assembler.py    # 256×32-bit 二进制编码
│   │   └── simulator.py    # Python 行为级 NPU 黄金模型
│   └── backends/
│       └── npu.py          # NPU Backend
├── tests/                  # 96 个测试
└── pyproject.toml          # ruff + mypy + pytest
```

## 开发

```bash
pip install -e ".[dev]"
pytest                    # 96 passed, 1 skipped
ruff check compiler/ tests/
mypy compiler/
```
