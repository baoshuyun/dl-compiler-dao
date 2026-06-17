// npu_top.v -- NPU SoC Top-Level Integration
// Part of AI_Compiler_Project / NPU_Project
//
// Architecture (data path):
//   Host -> Decoder (ISA instructions)
//            |-> DMA Engine (external DRAM <-> SRAM)
//            |-> Multi-Bank SRAM (Ping/Pong double buffer)
//            |-> Systolic Array (4x4 PEs)
//
// SRAM banks: bank0/2 = matrix A (ping/pong), bank1/3 = matrix B (ping/pong)
// Decoder executes pre-loaded instructions from instr_mem
// DMA handles data movement, Array handles computation
// BARRIER synchronizes DMA and Compute pipelines

`include "isa/isa_defines.vh"

module npu_top #(
    parameter DATA_WIDTH  = 32,
    parameter ADDR_WIDTH  = 10,
    parameter INSTR_DEPTH = 256,
    parameter ARRAY_ROWS  = 4,
    parameter ARRAY_COLS  = 4,
    parameter NUM_BANKS   = 4,
    parameter BANK_DEPTH  = 1024,
    parameter LATENCY     = ARRAY_ROWS + ARRAY_COLS - 1  // systolic pipeline cycles
) (
    input  wire                     clk,
    input  wire                     rst_n,
    // Host interface
    input  wire                     host_start,
    input  wire [31:0]              instr_mem [0:INSTR_DEPTH-1],
    output wire                     npu_done,
    // External memory interface (DRAM)
    output wire                     ext_req,
    input  wire                     ext_grant,
    output wire                     ext_rw,
    output wire [ADDR_WIDTH-1:0]    ext_addr,
    output wire [DATA_WIDTH-1:0]    ext_wdata,
    input  wire [DATA_WIDTH-1:0]    ext_rdata,
    input  wire                     ext_valid,
    // Debug output (all PE results)
    output wire [DATA_WIDTH-1:0]    debug_results [0:ARRAY_ROWS*ARRAY_COLS-1]
);

    // -- Internal wiring --
    // Decoder -> DMA
    wire                dec_dma_valid;
    wire                dec_dma_is_load;
    wire [1:0]          dec_dma_bank;
    wire [ADDR_WIDTH-1:0] dec_dma_ext_addr;
    wire [ADDR_WIDTH-1:0] dec_dma_size;
    wire                dma_busy;

    // Decoder -> Compute
    wire                dec_cmp_start;
    wire [1:0]          dec_cmp_mat_dim;
    wire [1:0]          dec_cmp_act;
    wire                cmp_busy;

    // DMA -> SRAM
    wire                dma_sram_wr_en;
    wire [1:0]          dma_sram_wr_bank;
    wire [ADDR_WIDTH-1:0] dma_sram_wr_addr;
    wire [DATA_WIDTH-1:0] dma_sram_wr_data;
    wire                dma_sram_rd_en;
    wire [1:0]          dma_sram_rd_bank;
    wire [ADDR_WIDTH-1:0] dma_sram_rd_addr;
    wire [DATA_WIDTH-1:0] dma_sram_rd_data;

    // SRAM -> Systolic Array
    wire [DATA_WIDTH-1:0] sram_pe_a_data;
    wire [DATA_WIDTH-1:0] sram_pe_b_data;

    // Systolic Array results
    wire [DATA_WIDTH-1:0] sa_results [0:ARRAY_ROWS*ARRAY_COLS-1];

    // Compute busy flag (pulsed by start, held until pipeline drains)
    localparam CYCLE_CNT_W = $clog2(LATENCY + 1);
    localparam DATA_CNT_W  = $clog2(ARRAY_ROWS);
    reg [CYCLE_CNT_W-1:0] cmp_cycle_cnt;
    reg [DATA_CNT_W-1:0]  cmp_data_cnt;
    reg       cmp_active;
    assign    cmp_busy = cmp_active;

    // SRAM read address counter: increments each cycle during compute
    wire [ADDR_WIDTH-1:0] pe_rd_a_addr_w;
    wire [ADDR_WIDTH-1:0] pe_rd_b_addr_w;
    assign pe_rd_a_addr_w = cmp_active ? cmp_data_cnt : {ADDR_WIDTH{1'b0}};
    assign pe_rd_b_addr_w = cmp_active ? cmp_data_cnt : {ADDR_WIDTH{1'b0}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmp_active    <= 1'b0;
            cmp_cycle_cnt <= {CYCLE_CNT_W{1'b0}};
            cmp_data_cnt  <= {DATA_CNT_W{1'b0}};
        end else begin
            if (dec_cmp_start) begin
                cmp_active    <= 1'b1;
                cmp_cycle_cnt <= {CYCLE_CNT_W{1'b0}};
                cmp_data_cnt  <= {DATA_CNT_W{1'b0}};
            end else if (cmp_active) begin
                if (cmp_cycle_cnt >= LATENCY[CYCLE_CNT_W-1:0]) begin
                    cmp_active    <= 1'b0;
                end else begin
                    cmp_cycle_cnt <= cmp_cycle_cnt + 1'b1;
                    // Stream new data during pipeline fill phase
                    if (cmp_data_cnt < (ARRAY_ROWS - 1))
                        cmp_data_cnt <= cmp_data_cnt + 1'b1;
                end
            end
        end
    end

    // -- Module Instantiations --

    decoder #(
        .INSTR_WIDTH(32),
        .INSTR_DEPTH(INSTR_DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_decoder (
        .clk              (clk),
        .rst_n            (rst_n),
        .host_start       (host_start),
        .instr_mem        (instr_mem),
        .dma_cmd_valid    (dec_dma_valid),
        .dma_cmd_is_load  (dec_dma_is_load),
        .dma_cmd_bank     (dec_dma_bank),
        .dma_cmd_ext_addr (dec_dma_ext_addr),
        .dma_cmd_size     (dec_dma_size),
        .dma_busy         (dma_busy),
        .cmp_start        (dec_cmp_start),
        .cmp_mat_dim      (dec_cmp_mat_dim),
        .cmp_act          (dec_cmp_act),
        .cmp_busy         (cmp_busy),
        .loop_counter     (),
        .loop_active      ()
    );

    dma_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_dma (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (dec_dma_valid),
        .cmd_is_load   (dec_dma_is_load),
        .cmd_bank      (dec_dma_bank),
        .cmd_ext_addr  (dec_dma_ext_addr),
        .cmd_size      (dec_dma_size),
        .sram_wr_en    (dma_sram_wr_en),
        .sram_wr_bank  (dma_sram_wr_bank),
        .sram_wr_addr  (dma_sram_wr_addr),
        .sram_wr_data  (dma_sram_wr_data),
        .sram_rd_data  (dma_sram_rd_data),
        .sram_rd_en    (dma_sram_rd_en),
        .sram_rd_bank  (dma_sram_rd_bank),
        .sram_rd_addr  (dma_sram_rd_addr),
        .ext_req       (ext_req),
        .ext_grant     (ext_grant),
        .ext_rw        (ext_rw),
        .ext_addr      (ext_addr),
        .ext_wdata     (ext_wdata),
        .ext_rdata     (ext_rdata),
        .ext_valid     (ext_valid),
        .dma_busy      (dma_busy)
    );

    multi_bank_sram #(
        .NUM_BANKS(NUM_BANKS),
        .BANK_DEPTH(BANK_DEPTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_sram (
        .clk            (clk),
        .rst_n          (rst_n),
        .dma_wr_en      (dma_sram_wr_en),
        .dma_wr_bank    (dma_sram_wr_bank),
        .dma_wr_addr    (dma_sram_wr_addr),
        .dma_wr_data    (dma_sram_wr_data),
        .pe_rd_a_en     (1'b1),
        .pe_rd_a_bank   (2'b00),           // bank 0 = matrix A ping (can be made dynamic)
        .pe_rd_a_addr   (pe_rd_a_addr_w),
        .pe_rd_a_data   (sram_pe_a_data),
        .pe_rd_b_en     (1'b1),
        .pe_rd_b_bank   (2'b01),           // bank 1 = matrix B ping (can be made dynamic)
        .pe_rd_b_addr   (pe_rd_b_addr_w),
        .pe_rd_b_data   (sram_pe_b_data)
    );

    // SRAM → Systolic Array: fan-out one read port to all rows/cols
    wire [DATA_WIDTH-1:0] sa_a_data [0:ARRAY_ROWS-1];
    wire [DATA_WIDTH-1:0] sa_b_data [0:ARRAY_COLS-1];
    genvar gi;
    generate
        for (gi = 0; gi < ARRAY_ROWS; gi = gi + 1)
            assign sa_a_data[gi] = sram_pe_a_data;
        for (gi = 0; gi < ARRAY_COLS; gi = gi + 1)
            assign sa_b_data[gi] = sram_pe_b_data;
    endgenerate

    systolic_array #(
        .ROWS(ARRAY_ROWS),
        .COLS(ARRAY_COLS),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_array (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (dec_cmp_start),
        .stall    (!cmp_busy),    // Freeze PE after pipeline completes
        .mat_dim  (dec_cmp_mat_dim),
        .a_data   (sa_a_data),
        .b_data   (sa_b_data),
        .results  (sa_results)
    );

    assign debug_results = sa_results;
    // npu_done only valid after decoder has started running
    reg npu_started;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            npu_started <= 1'b0;
        else if (host_start)
            npu_started <= 1'b1;
    end
    assign npu_done = npu_started && !cmp_busy && !dma_busy && |cmp_cycle_cnt;

endmodule
