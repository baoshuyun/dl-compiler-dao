// dma_engine.v -- DMA Engine with Ping/Pong double buffering
// Part of AI_Compiler_Project / NPU_Project
//
// Architecture:
// - 6+1 state FSM: IDLE -> REQ -> WAIT_RD -> WR_SRAM -> WR_EXT -> DONE
// - Ping/Pong buffer selection: while LOAD to ping, COMPUTE from pong (and vice versa)
// - pp_sel bit selects ping (0) or pong (1) buffer
// - Bank addressing: cmd_bank = {pp_sel, ab_sel}
//   ab_sel=0: bank0/2 (matrix A), ab_sel=1: bank1/3 (matrix B)
// - Pending buffer: if a new command arrives while DMA is busy, store it
//   and execute immediately after current transfer completes

module dma_engine #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 16,
    parameter BURST_LEN  = 8
) (
    input  wire                     clk,
    input  wire                     rst_n,
    // Command interface (from decoder)
    input  wire                     cmd_valid,
    input  wire                     cmd_is_load,    // 1=LOAD (ext->SRAM), 0=STORE (SRAM->ext)
    input  wire [1:0]               cmd_bank,       // Target SRAM bank
    input  wire [ADDR_WIDTH-1:0]    cmd_ext_addr,   // External memory address
    input  wire [ADDR_WIDTH-1:0]    cmd_size,       // Transfer size in words
    // SRAM interface
    output reg                      sram_wr_en,
    output reg  [1:0]               sram_wr_bank,
    output reg  [ADDR_WIDTH-1:0]    sram_wr_addr,
    output reg  [DATA_WIDTH-1:0]    sram_wr_data,
    input  wire [DATA_WIDTH-1:0]    sram_rd_data,
    output reg                      sram_rd_en,
    output reg  [1:0]               sram_rd_bank,
    output reg  [ADDR_WIDTH-1:0]    sram_rd_addr,
    // External memory interface (AXI-lite simplified)
    output reg                      ext_req,
    input  wire                     ext_grant,
    output reg                      ext_rw,         // 1=read, 0=write
    output reg  [ADDR_WIDTH-1:0]    ext_addr,
    output reg  [DATA_WIDTH-1:0]    ext_wdata,
    input  wire [DATA_WIDTH-1:0]    ext_rdata,
    input  wire                     ext_valid,
    // Status
    output wire                     dma_busy
);

    // -- FSM State Definitions --
    localparam [2:0]
        IDLE    = 3'd0,
        REQ     = 3'd1,
        WAIT_RD = 3'd2,
        WR_SRAM = 3'd3,
        WR_EXT  = 3'd4,
        DONE    = 3'd5;

    reg [2:0] state, next_state;

    // Pending command buffer (for pipelining)
    reg             pend_valid;
    reg             pend_is_load;
    reg [1:0]       pend_bank;
    reg [ADDR_WIDTH-1:0] pend_ext_addr;
    reg [ADDR_WIDTH-1:0] pend_size;

    // Active transfer registers
    reg             is_load;
    reg [1:0]       dst_bank;
    reg [ADDR_WIDTH-1:0] xfer_cnt;
    reg [ADDR_WIDTH-1:0] ext_addr_reg;
    reg [ADDR_WIDTH-1:0] sram_addr_reg;

    // DMA busy = not idle OR pending command waiting
    assign dma_busy = (state != IDLE) || pend_valid;

    // -- FSM Sequential Logic --
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            pend_valid  <= 1'b0;
            is_load     <= 1'b0;
            dst_bank    <= 2'b00;
            xfer_cnt    <= {ADDR_WIDTH{1'b0}};
            ext_addr_reg <= {ADDR_WIDTH{1'b0}};
            sram_addr_reg <= {ADDR_WIDTH{1'b0}};
            sram_wr_en  <= 1'b0;
            sram_rd_en  <= 1'b0;
            ext_req     <= 1'b0;
        end else begin
            state <= next_state;

            case (state)
                IDLE: begin
                    sram_wr_en <= 1'b0;
                    sram_rd_en <= 1'b0;
                    ext_req    <= 1'b0;

                    if (cmd_valid) begin
                        is_load      <= cmd_is_load;
                        dst_bank     <= cmd_bank;
                        ext_addr_reg <= cmd_ext_addr;
                        xfer_cnt     <= cmd_size;
                        sram_addr_reg <= {ADDR_WIDTH{1'b0}};
                        pend_valid   <= 1'b0;
                    end else if (pend_valid) begin
                        // Process queued pending command
                        is_load      <= pend_is_load;
                        dst_bank     <= pend_bank;
                        ext_addr_reg <= pend_ext_addr;
                        xfer_cnt     <= pend_size;
                        sram_addr_reg <= {ADDR_WIDTH{1'b0}};
                        pend_valid   <= 1'b0;
                    end
                end

                REQ: begin
                    ext_req <= 1'b1;
                    ext_rw  <= is_load;  // 1=read
                    ext_addr <= ext_addr_reg;
                end

                WAIT_RD: begin
                    ext_req <= 1'b0;
                    if (ext_valid) begin
                        if (is_load) begin
                            // Write received data to SRAM
                            sram_wr_en   <= 1'b1;
                            sram_wr_bank <= dst_bank;
                            sram_wr_addr <= sram_addr_reg;
                            sram_wr_data <= ext_rdata;
                        end else begin
                            // Read data from SRAM to send out
                            sram_rd_en   <= 1'b1;
                            sram_rd_bank <= dst_bank;
                            sram_rd_addr <= sram_addr_reg;
                        end
                    end
                end

                WR_SRAM: begin
                    sram_wr_en <= 1'b0;
                    sram_addr_reg <= sram_addr_reg + 1'b1;
                    ext_addr_reg  <= ext_addr_reg + 1'b1;
                    if (sram_addr_reg >= xfer_cnt - 1) begin
                        // Transfer complete
                        sram_wr_en <= 1'b0;
                    end
                end

                WR_EXT: begin
                    sram_rd_en <= 1'b0;
                    ext_wdata  <= sram_rd_data;
                    sram_addr_reg <= sram_addr_reg + 1'b1;
                    ext_addr_reg  <= ext_addr_reg + 1'b1;
                end

                DONE: begin
                    ext_req <= 1'b0;
                end
            endcase

            // Queue incoming commands while busy
            if (cmd_valid && state != IDLE && !pend_valid) begin
                pend_valid    <= 1'b1;
                pend_is_load  <= cmd_is_load;
                pend_bank     <= cmd_bank;
                pend_ext_addr <= cmd_ext_addr;
                pend_size     <= cmd_size;
            end
        end
    end

    // -- FSM Next State Logic --
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:    if (cmd_valid || pend_valid) next_state = REQ;
            REQ:     if (ext_grant)               next_state = WAIT_RD;
            WAIT_RD: if (ext_valid) begin
                         if (is_load) next_state = WR_SRAM;
                         else          next_state = WR_EXT;
                     end
            WR_SRAM: if (sram_addr_reg >= xfer_cnt - 1) next_state = DONE;
                     else                                next_state = REQ;  // Next word
            WR_EXT:  if (sram_addr_reg >= xfer_cnt - 1) next_state = DONE;
                     else                                next_state = REQ;
            DONE:    next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

endmodule
