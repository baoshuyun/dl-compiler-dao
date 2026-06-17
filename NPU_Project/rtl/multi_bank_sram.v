// multi_bank_sram.v -- Multi-Bank SRAM with dual read + single write ports
// Part of AI_Compiler_Project / NPU_Project
//
// Architecture:
// - 4 banks (2 for matrix A, 2 for matrix B) for Ping/Pong double buffering
// - 1 write port (from DMA) + 2 read ports (to systolic array: A and B)
// - Each bank: independent address space, byte-addressable
// - Read ports feed PE array inputs simultaneously

module multi_bank_sram #(
    parameter NUM_BANKS    = 4,
    parameter BANK_DEPTH   = 1024,     // 1K words per bank
    parameter DATA_WIDTH   = 32,
    parameter ADDR_WIDTH   = 10        // log2(BANK_DEPTH)
) (
    input  wire                     clk,
    input  wire                     rst_n,
    // DMA write port (1 write per cycle)
    input  wire                     dma_wr_en,
    input  wire [BANK_SEL_W-1:0]    dma_wr_bank,
    input  wire [ADDR_WIDTH-1:0]    dma_wr_addr,
    input  wire [DATA_WIDTH-1:0]    dma_wr_data,
    // PE read port A (for weights / left matrix)
    input  wire                     pe_rd_a_en,
    input  wire [BANK_SEL_W-1:0]    pe_rd_a_bank,
    input  wire [ADDR_WIDTH-1:0]    pe_rd_a_addr,
    output wire [DATA_WIDTH-1:0]    pe_rd_a_data,
    // PE read port B (for activations / right matrix)
    input  wire                     pe_rd_b_en,
    input  wire [BANK_SEL_W-1:0]    pe_rd_b_bank,
    input  wire [ADDR_WIDTH-1:0]    pe_rd_b_addr,
    output wire [DATA_WIDTH-1:0]    pe_rd_b_data
);

    localparam BANK_SEL_W = (NUM_BANKS > 1) ? $clog2(NUM_BANKS) : 1;

    // -- Bank storage --
    reg [DATA_WIDTH-1:0] bank [0:NUM_BANKS-1][0:BANK_DEPTH-1];

    // -- Write logic (synchronous) --
    integer bank_idx;
    always @(posedge clk) begin
        if (dma_wr_en) begin
            bank[dma_wr_bank][dma_wr_addr] <= dma_wr_data;
        end
    end

    // -- Read logic (combinational for low latency) --
    assign pe_rd_a_data = (pe_rd_a_en) ? bank[pe_rd_a_bank][pe_rd_a_addr] : {DATA_WIDTH{1'b0}};
    assign pe_rd_b_data = (pe_rd_b_en) ? bank[pe_rd_b_bank][pe_rd_b_addr] : {DATA_WIDTH{1'b0}};

endmodule
