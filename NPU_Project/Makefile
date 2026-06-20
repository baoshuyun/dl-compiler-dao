# NPU_Project — Verilog RTL simulation
# Usage: make sim

IVERILOG = iverilog
VVP = vvp
RTL_DIR = rtl
SIM_DIR = sim

RTL_SRC = $(RTL_DIR)/pe.v \
          $(RTL_DIR)/systolic_array.v \
          $(RTL_DIR)/dma_engine.v \
          $(RTL_DIR)/multi_bank_sram.v \
          $(RTL_DIR)/decoder.v \
          $(RTL_DIR)/npu_top.v

TB_SRC = $(SIM_DIR)/tb_npu_top.v
OUT = $(SIM_DIR)/tb_npu_top

.PHONY: sim clean

sim: $(OUT)
	$(VVP) $(OUT)

$(OUT): $(RTL_SRC) $(TB_SRC)
	$(IVERILOG) -g2012 -I $(RTL_DIR) -I ../isa -o $@ $(RTL_SRC) $(TB_SRC)

clean:
	rm -f $(OUT) $(SIM_DIR)/*.vcd
