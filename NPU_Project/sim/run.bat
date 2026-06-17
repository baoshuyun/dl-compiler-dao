@echo off
:: NPU_Project run script (Windows CMD)
cd /d %~dp0
echo Compiling...
C:\iverilog\bin\iverilog -g2012 -I .. -o tb_npu_top ../rtl/pe.v ../rtl/systolic_array.v ../rtl/dma_engine.v ../rtl/multi_bank_sram.v ../rtl/decoder.v ../rtl/npu_top.v tb_npu_top.v
if %errorlevel% neq 0 exit /b %errorlevel%
echo Running...
C:\iverilog\bin\vvp tb_npu_top
