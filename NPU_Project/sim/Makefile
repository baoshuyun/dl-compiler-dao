# NPU_Project Makefile
# Usage: make        (compile + run)
#        make compile (compile only)
#        make run     (run only)
#        make wave    (open VCD waveform)
#        make clean   (remove generated files)

RTL = ../rtl
SIM = .

SRC = $(RTL)/pe.v $(RTL)/systolic_array.v $(RTL)/dma_engine.v \
      $(RTL)/multi_bank_sram.v $(RTL)/decoder.v $(RTL)/npu_top.v \
      $(SIM)/tb_npu_top.v

OUT = tb_npu_top
VCD = tb_npu_top.vcd

all: compile run

compile:
	iverilog -g2012 -I .. -o $(OUT) $(SRC)

run:
	vvp $(OUT)

wave:
	gtkwave $(VCD) &

clean:
	rm -f $(OUT) $(VCD)
