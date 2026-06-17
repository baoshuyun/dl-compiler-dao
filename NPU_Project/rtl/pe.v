// pe.v -- Processing Element: MAC (Multiply-Accumulate) with pipeline registers
// Part of AI_Compiler_Project / NPU_Project
//
// Each PE performs: p_out += a_in * b_in
// Data flows: a_in -> a_out (horizontal, to right neighbor)
//             b_in -> b_out (vertical, to bottom neighbor)
//             p_out accumulates locally
// Stall signal gates data flow for systolic pipeline control.

module pe #(
    parameter DATA_WIDTH = 32  // FP32
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,      // Reset accumulator to 0
    input  wire                 stall,      // Stall data propagation
    input  wire [DATA_WIDTH-1:0] a_in,
    input  wire [DATA_WIDTH-1:0] b_in,
    input  wire                 a_valid,
    input  wire                 b_valid,
    output reg  [DATA_WIDTH-1:0] a_out,
    output reg  [DATA_WIDTH-1:0] b_out,
    output reg  [DATA_WIDTH-1:0] p_out,
    output reg                  p_valid
);

    // Internal accumulation register
    reg [DATA_WIDTH-1:0] acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_out   <= 32'd0;
            p_valid <= 1'b0;
            a_out   <= 32'd0;
            b_out   <= 32'd0;
            acc     <= 32'd0;
        end else if (start) begin
            // Reset accumulator at start of new computation
            p_out   <= 32'd0;
            p_valid <= 1'b0;
            acc     <= 32'd0;
            a_out   <= a_in;
            b_out   <= b_in;
        end else if (!stall) begin
            // Forward inputs to outputs (systolic data movement)
            a_out <= a_in;
            b_out <= b_in;

            if (a_valid && b_valid) begin
                // Integer MAC (for behavioral simulation)
                acc   <= acc + a_in * b_in;
                p_out <= acc + a_in * b_in;
                p_valid <= 1'b1;
            end else begin
                p_valid <= 1'b0;
            end
        end
    end

endmodule
