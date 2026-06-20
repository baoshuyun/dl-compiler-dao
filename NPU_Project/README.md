# NPU_Project · 卷叁
光靠软件慢慢算太慢了，这个项目就是专门造那口真正能炒菜的“铁锅”。
**但最值钱的是那个“防乱翻开关”（BARRIER）：**
算得快不算本事，该刹得住才是真本事。不乱序、不乱翻，这口铁锅才能一直稳下去。

> Verilog RTL · 参数化脉动阵列 · 自定义 32-bit ISA  
> 与 mini-dl-compiler / NPU_Soft_Hard_Stack 底层互通 (共用 isa.py ↔ isa_defines.vh)

## 架构

```
Host CPU
  │
  ▼
Decoder (ISA 指令解码 + PC + LOOP/BARRIER)
  │
  ├──→ DMA Engine (6-FSM, Ping/Pong 双缓冲, 挂起命令队列)
  │      └──→ Multi-Bank SRAM (NUM_BANKS × BANK_DEPTH, 参数化)
  │             └──→ Systolic Array (ROWS×COLS PE, 权重驻留)
  └──→ (COMPUTE 控制信号)
```

## ISA 指令集 (32-bit 定长, 与 isa.py 同步)

| Opcode | 助记符 | 编码 |
|--------|--------|------|
| 0x0 | NOP | — |
| 0x1 | LOAD | `bank[27:26] size[25:16] addr[15:0]` |
| 0x2 | STORE | `bank[27:26] size[25:16] addr[15:0]` |
| 0x3 | COMPUTE | `a[27:26] b[25:24] c[23:22] dim[21:20] act[19:18]` |
| 0x4 | BARRIER | `type[27:26]` (01=DMA, 10=COMPUTE, 11=ALL) |
| 0x5 | CONFIG | (保留) |
| 0x6 | LOOP | `count[15:0]` |

## 参数 (完全参数化)

| 参数 | 默认 | 说明 |
|------|------|------|
| ARRAY_ROWS | 4 | PE 行数 |
| ARRAY_COLS | 4 | PE 列数 |
| NUM_BANKS | 4 | SRAM Bank 数 |
| BANK_DEPTH | 1024 | 每 Bank 深度 (32-bit words) |
| LATENCY | ROWS+COLS-1 | 流水线周期 |
| DATA_WIDTH | 32 | 数据位宽 |

## 项目结构

```
NPU_Project/
├── isa/
│   └── isa_defines.vh        # ISA 编码 (与 compiler/isa_mapper/isa.py 同步)
├── rtl/
│   ├── npu_top.v             # SoC 顶层 (参数化: ROWS/COLS/BANKS/DEPTH)
│   ├── decoder.v             # 指令解码 + PC + LOOP/BARRIER 实现
│   ├── dma_engine.v          # 6-FSM + Ping/Pong + 挂起队列
│   ├── multi_bank_sram.v     # 参数化 Bank 数 + 深度, 双读口
│   ├── systolic_array.v      # 参数化 PE 网格 + 子阵列 (4×4/2×2/1×1)
│   └── pe.v                  # MAC: acc += a × b
├── sim/
│   ├── tb_npu_top.v          # 自检测试台 (恒等矩阵)
│   ├── test_utils.py         # Python 黄金模型 + 随机矩阵 + 指令生成
│   ├── Makefile              # 编译 + 仿真 + 波形
│   └── run.bat / run.sh
├── constraints/
│   └── npu_top.sdc           # 100MHz SDC 综合约束
├── docs/
│   └── register_map.md       # 寄存器映射 + 编程指南
└── README.md
```

## 仿真

```bash
cd sim
make          # iverilog 编译 + 运行
make wave     # GTKWave 查看波形
```

## 已验证的 Issue (2026-06-05 已修复)

1. ISA 编码: COMPUTE/BARRIER 保留字段宽度修正
2. Decoder 流水线停顿: BARRIER stall 时 `current_instr` 过早覆盖
3. 外部内存握手: `ext_grant`/`ext_valid` 改为组合逻辑
4. PE MAC: 从 FP32 位模式整数乘 → 行为级整数运算
5. `npu_done`: 增加 `npu_started` 防复位误触发
6. SRAM 读地址: 从固定 0 → 循环计数器流式送数
7. PE stall: `!cmp_busy` 冻结流水线
8. 参数化: 全部硬编码尺寸改为 `parameter` (2026-06-14)

## 与软件栈互通

```
Soft_Stack NPUBackend ──┐
                        ├──→ ISAMapper → 256×32-bit instr_mem
mini-dl ISAMapper ──────┘         │
                                   ▼
                          NPU_Project.instr_mem[0:255]
```
