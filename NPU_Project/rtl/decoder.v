// decoder.v -- Instruction Decoder with Program Counter
// Part of AI_Compiler_Project / NPU_Project
//
// Decodes 32-bit ISA instructions and drives control signals to:
// - DMA engine (LOAD/STORE commands)
// - Systolic array (COMPUTE with mat_dim and activation)
// - Barrier synchronization logic
// - Configuration registers
// - Loop counter for repeated instruction sequences
//
// BARRIER implementation: PC stalls until the specified engine is idle.
//   BAR_DMA:       while (dma_busy) wait;
//   BAR_COMPUTE:   while (cmp_busy) wait;
//   BAR_ALL:       while (dma_busy || cmp_busy) wait;

`include "../isa/isa_defines.vh"

module decoder #(
    parameter INSTR_WIDTH = 32,
    parameter INSTR_DEPTH = 256,     // Instruction memory depth
    parameter ADDR_WIDTH  = 8        // log2(INSTR_DEPTH)
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     host_start,    // Host triggers execution
    // Instruction memory interface (pre-loaded by host)
    input  wire [INSTR_WIDTH-1:0]   instr_mem [0:INSTR_DEPTH-1],
    // DMA command interface
    output reg                      dma_cmd_valid,
    output reg                      dma_cmd_is_load,
    output reg  [1:0]               dma_cmd_bank,
    output reg  [ADDR_WIDTH-1:0]    dma_cmd_ext_addr,
    output reg  [ADDR_WIDTH-1:0]    dma_cmd_size,
    input  wire                     dma_busy,
    // Compute command interface
    output reg                      cmp_start,
    output reg  [1:0]               cmp_mat_dim,
    output reg  [1:0]               cmp_act,
    input  wire                     cmp_busy,
    // Loop counter
    output reg  [15:0]              loop_counter,
    output reg                      loop_active
);

    // Program Counter
    reg [ADDR_WIDTH-1:0] pc;
    reg [INSTR_WIDTH-1:0] current_instr;
    wire [3:0] opcode;
    assign opcode = current_instr[31:28];

    // Loop state
    reg [15:0] loop_count;
    reg [ADDR_WIDTH-1:0] loop_start_pc;

    // Running flag
    reg running;
    reg instr_done;  // High when current instruction completed, load next
    reg load_next;   // Combinational-style flag for immediate evaluation

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc             <= {ADDR_WIDTH{1'b0}};
            running        <= 1'b0;
            instr_done     <= 1'b1;
            loop_active    <= 1'b0;
            loop_counter   <= 16'd0;
            loop_count     <= 16'd0;
            loop_start_pc  <= {ADDR_WIDTH{1'b0}};
            dma_cmd_valid  <= 1'b0;
            cmp_start      <= 1'b0;
        end else begin
            if (host_start && !running) begin
                running <= 1'b1;
                pc      <= {ADDR_WIDTH{1'b0}};
            end

            if (running) begin
                // Clear one-shot signals
                dma_cmd_valid <= 1'b0;
                cmp_start     <= 1'b0;

                // Use blocking assignment for immediate case override
                load_next = 1'b1;

                case (opcode)
                    `OP_NOP: begin
                        // No operation, PC advances automatically
                    end

                    `OP_LOAD: begin
                        dma_cmd_valid    <= 1'b1;
                        dma_cmd_is_load  <= 1'b1;
                        dma_cmd_bank     <= current_instr[27:26];
                        dma_cmd_ext_addr <= current_instr[15:0];
                        dma_cmd_size     <= current_instr[25:16];
                    end

                    `OP_STORE: begin
                        dma_cmd_valid    <= 1'b1;
                        dma_cmd_is_load  <= 1'b0;
                        dma_cmd_bank     <= current_instr[27:26];
                        dma_cmd_ext_addr <= current_instr[15:0];
                        dma_cmd_size     <= current_instr[25:16];
                    end

                    `OP_COMPUTE: begin
                        cmp_start   <= 1'b1;
                        cmp_mat_dim <= current_instr[21:20];
                        cmp_act     <= current_instr[19:18];
                    end

                    `OP_BARRIER: begin
                        case (current_instr[1:0])
                            `BAR_DMA:     if (dma_busy)             load_next = 1'b0;
                            `BAR_COMPUTE: if (cmp_busy)             load_next = 1'b0;
                            `BAR_ALL:     if (dma_busy || cmp_busy) load_next = 1'b0;
                        endcase
                    end

                    `OP_CONFIG: begin
                        // Configuration value in lower 16 bits
                        // PC advances normally
                    end

                    `OP_LOOP: begin
                        if (!loop_active) begin
                            // Start loop: next instr is loop body
                            loop_active   <= 1'b1;
                            loop_count    <= current_instr[15:0];
                            loop_counter  <= current_instr[15:0];
                            loop_start_pc <= pc + 1'b1;
                        end else if (loop_counter > 16'd1) begin
                            // Continue looping: jump back
                            loop_counter <= loop_counter - 16'd1;
                            pc           <= loop_start_pc;
                            load_next    = 1'b0;  // Jump, don't auto-advance
                        end else begin
                            // Loop done
                            loop_active   <= 1'b0;
                            loop_counter  <= 16'd0;
                        end
                    end

                    default: begin
                        // Unknown opcode: treat as NOP
                    end
                endcase

                // Load next instruction if current one completed (not stalled)
                if (load_next) begin
                    instr_done   <= 1'b1;
                    current_instr <= instr_mem[pc];
                    pc <= pc + 1'b1;
                end else begin
                    instr_done <= 1'b0;
                end

                // Auto-halt when PC exceeds instruction memory
                if (pc >= INSTR_DEPTH - 1) begin
                    running <= 1'b0;
                end
            end
        end
    end

endmodule
