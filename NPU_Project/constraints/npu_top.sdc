# Synopsys Design Constraints for NPU Top
# Target: 7-series FPGA (Artix-7 / Kintex-7) at 100 MHz
#
# Usage:
#   Vivado:  add_files -fileset constrs_1 npu_top.sdc
#   Yosys:   read_sdc npu_top.sdc

# ── Clock ──────────────────────────────────────────────────────
create_clock -name clk -period 10.0 [get_ports clk]

# ── Input delays (external DRAM / host interface) ───────────────
set_input_delay -clock clk -max 3.0 [get_ports {host_start}]
set_input_delay -clock clk -min 1.0 [get_ports {host_start}]

set_input_delay -clock clk -max 4.0 [get_ports {instr_mem[*]}]
set_input_delay -clock clk -min 1.5 [get_ports {instr_mem[*]}]

set_input_delay -clock clk -max 4.0 [get_ports {ext_rdata[*]}]
set_input_delay -clock clk -min 1.5 [get_ports {ext_rdata[*]}]

set_input_delay -clock clk -max 3.0 [get_ports {ext_grant}]
set_input_delay -clock clk -min 1.0 [get_ports {ext_grant}]

set_input_delay -clock clk -max 3.0 [get_ports {ext_valid}]
set_input_delay -clock clk -min 1.0 [get_ports {ext_valid}]

# ── Output delays ───────────────────────────────────────────────
set_output_delay -clock clk -max 3.0 [get_ports {npu_done}]
set_output_delay -clock clk -min 1.0 [get_ports {npu_done}]

set_output_delay -clock clk -max 4.0 [get_ports {ext_req}]
set_output_delay -clock clk -min 1.5 [get_ports {ext_req}]

set_output_delay -clock clk -max 4.0 [get_ports {ext_rw}]
set_output_delay -clock clk -min 1.5 [get_ports {ext_rw}]

set_output_delay -clock clk -max 4.0 [get_ports {ext_addr[*]}]
set_output_delay -clock clk -min 1.5 [get_ports {ext_addr[*]}]

set_output_delay -clock clk -max 4.0 [get_ports {ext_wdata[*]}]
set_output_delay -clock clk -min 1.5 [get_ports {ext_wdata[*]}]

set_output_delay -clock clk -max 4.0 [get_ports {debug_results[*]}]
set_output_delay -clock clk -min 1.5 [get_ports {debug_results[*]}]

# ── False paths (async reset) ──────────────────────────────────
set_false_path -from [get_ports rst_n]

# ── Multicycle paths ────────────────────────────────────────────
# Systolic array pipeline: PE-to-PE propagation is single-cycle
# but the overall latency is ROWS+COLS-1 cycles.
# No explicit multicycle needed; the design self-synchronizes
# via start/stall.

# ── Clock groups ────────────────────────────────────────────────
# Single clock domain — no cross-domain constraints needed.
