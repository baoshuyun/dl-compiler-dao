// systolic_array.v -- 4x4 Weight-Stationary Systolic Array
// Part of AI_Compiler_Project / NPU_Project
//
// Architecture:
// - 4x4 grid of PEs (16 total) with staggered pipeline
// - 7 cycles total latency = ROWS + COLS - 1
// - Cycles 0-3: pipeline fill (data flows into array)
// - Cycles 4-6: pipeline drain (results flow out)
// - Cycle 7: all PEs have valid results
// - Supports 2x2 and 1x1 sub-array modes via row_en/col_en
//
// Dataflow:
//   A (weights) flows left-to-right (horizontal)
//   B (activations) flows top-to-bottom (vertical)
//   C (partial sums) accumulates in-place at each PE

`include "../isa/isa_defines.vh"

module systolic_array #(
    parameter ROWS = 4,
    parameter COLS = 4,
    parameter DATA_WIDTH = 32,
    parameter LATENCY = ROWS + COLS - 1  // = 7 for 4x4
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,       // Begin new computation
    input  wire                     stall,       // Stall pipeline
    input  wire [1:0]               mat_dim,     // MAT_4x4 / MAT_2x2 / MAT_1x1
    input  wire [DATA_WIDTH-1:0]   a_data [0:ROWS-1],   // Pre-loaded weights (rows)
    input  wire [DATA_WIDTH-1:0]   b_data [0:COLS-1],   // Streaming activations (cols)
    output wire [DATA_WIDTH-1:0]   results [0:ROWS*COLS-1]  // PE outputs
);

    // -- Inter-PE wiring (systolic mesh) --
    // h_wire[r][c] = data entering PE[r][c] from the left
    //   h_wire[r][0] = a_data[r] (external input)
    //   h_wire[r][c+1] = PE[r][c].a_out (c from 0 to COLS-1)
    wire [DATA_WIDTH-1:0] h_wire [0:ROWS-1][0:COLS];

    // v_wire[r][c] = data entering PE[r][c] from the top
    //   v_wire[0][c] = b_data[c] (external input)
    //   v_wire[r+1][c] = PE[r][c].b_out (r from 0 to ROWS-1)
    wire [DATA_WIDTH-1:0] v_wire [0:ROWS][0:COLS-1];

    // -- Connect external inputs to the first row/column of the mesh --
    genvar i;
    generate
        for (i = 0; i < ROWS; i = i + 1) assign h_wire[i][0] = a_data[i];
        for (i = 0; i < COLS; i = i + 1) assign v_wire[0][i] = b_data[i];
    endgenerate

    // -- Sub-array control (parameterized) --
    wire [ROWS-1:0] row_en;
    wire [COLS-1:0] col_en;

    // Build enable masks from ROWS/COLS parameters
    function [ROWS-1:0] build_row_en;
        input [1:0] dim;
        integer i;
        begin
            build_row_en = {ROWS{1'b0}};
            for (i = 0; i < ROWS; i = i + 1) begin
                if (dim == `MAT_4x4 && i < 4) build_row_en[i] = 1'b1;
                if (dim == `MAT_2x2 && i < 2) build_row_en[i] = 1'b1;
                if (dim == `MAT_1x1 && i < 1) build_row_en[i] = 1'b1;
            end
        end
    endfunction

    function [COLS-1:0] build_col_en;
        input [1:0] dim;
        integer i;
        begin
            build_col_en = {COLS{1'b0}};
            for (i = 0; i < COLS; i = i + 1) begin
                if (dim == `MAT_4x4 && i < 4) build_col_en[i] = 1'b1;
                if (dim == `MAT_2x2 && i < 2) build_col_en[i] = 1'b1;
                if (dim == `MAT_1x1 && i < 1) build_col_en[i] = 1'b1;
            end
        end
    endfunction

    assign row_en = build_row_en(mat_dim);
    assign col_en = build_col_en(mat_dim);

    // -- Generate PE grid --
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : row_pe
            for (c = 0; c < COLS; c = c + 1) begin : col_pe
                pe #(.DATA_WIDTH(DATA_WIDTH)) u_pe (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .start   (start),
                    .stall   (stall || !row_en[r] || !col_en[c]),
                    .a_in    (h_wire[r][c]),        // from left (or a_data at col 0)
                    .b_in    (v_wire[r][c]),        // from top (or b_data at row 0)
                    .a_valid (row_en[r] && col_en[c]),
                    .b_valid (row_en[r] && col_en[c]),
                    .a_out   (h_wire[r][c+1]),      // to right neighbor
                    .b_out   (v_wire[r+1][c]),      // to bottom neighbor
                    .p_out   (results[r * COLS + c]),
                    .p_valid ()
                );
            end
        end
    endgenerate

endmodule
