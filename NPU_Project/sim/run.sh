#!/bin/bash
# NPU_Project run script
# Usage: ./run.sh

cd "$(dirname "$0")"
echo "Compiling..."
iverilog -g2012 -I .. -o tb_npu_top \
  ../rtl/pe.v ../rtl/systolic_array.v ../rtl/dma_engine.v \
  ../rtl/multi_bank_sram.v ../rtl/decoder.v ../rtl/npu_top.v \
  tb_npu_top.v || exit 1
echo "Running..."
vvp tb_npu_top
