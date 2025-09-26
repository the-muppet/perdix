/**
 * GPU-Parallel Lisp Parser and VM
 * 
 * Exploits S-expression structure for embarrassingly parallel parsing.
 * Fixed-size expressions allow perfect work distribution across threads.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// ============================================================================
// Fixed-Size S-Expression Tokens
// ============================================================================

#define MAX_TOKEN_LENGTH 32
#define MAX_EXPR_TOKENS 16  // Max tokens per expression
#define MAX_EXPR_SIZE 256   // Max bytes per expression

enum TokenType {
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_SYMBOL,
    TOK_NUMBER,
    TOK_STRING,
    TOK_EOF,
    TOK_ERROR
};

struct Token {
    TokenType type;
    union {
        float number;
        uint32_t symbol_id;
        uint32_t string_idx;
    } value;
    uint16_t source_offset;  // Where in source
    uint16_t length;
};

// Fixed-size expression for GPU processing
struct SExpression {
    Token tokens[MAX_EXPR_TOKENS];
    uint8_t token_count;
    uint8_t depth;  // Nesting depth
    uint8_t flags;
    uint8_t _pad;
};

// Symbol table (compile-time for simplicity)
__constant__ const char* SYMBOLS[] = {
    "define", "lambda", "if", "let", "begin", "set!",
    "draw", "clear", "color", "rect", "text", "viewport",
    "pipeline", "render-pass", "push-constants",
    "+", "-", "*", "/", "=", "<", ">",
    "and", "or", "not"
};

__constant__ uint32_t NUM_SYMBOLS = 24;

// ============================================================================
// Parallel S-Expression Finder
// ============================================================================

__device__ void find_expression_boundaries(
    const char* source,
    uint32_t source_len,
    uint32_t* expr_starts,
    uint32_t* expr_ends,
    uint32_t* expr_count,
    uint32_t tid,
    uint32_t num_threads
) {
    // Each thread scans a portion of source
    uint32_t chunk_size = (source_len + num_threads - 1) / num_threads;
    uint32_t start = tid * chunk_size;
    uint32_t end = min(start + chunk_size, source_len);
    
    int depth = 0;
    uint32_t expr_start = UINT32_MAX;
    
    for (uint32_t i = start; i < end; i++) {
        if (source[i] == '(') {
            if (depth == 0) {
                expr_start = i;
            }
            depth++;
        } else if (source[i] == ')') {
            depth--;
            if (depth == 0 && expr_start != UINT32_MAX) {
                // Found complete expression
                uint32_t idx = atomicAdd(expr_count, 1);
                if (idx < 1024) {  // Max expressions
                    expr_starts[idx] = expr_start;
                    expr_ends[idx] = i + 1;
                }
                expr_start = UINT32_MAX;
            }
        }
    }
}

// ============================================================================
// Parallel S-Expression Parser
// ============================================================================

__device__ uint32_t parse_symbol(const char* str, uint32_t len) {
    // Hash symbol to ID (simple hash for demo)
    uint32_t hash = 5381;
    for (uint32_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + str[i];
    }
    
    // Check against known symbols
    for (uint32_t i = 0; i < NUM_SYMBOLS; i++) {
        bool match = true;
        const char* sym = SYMBOLS[i];
        for (uint32_t j = 0; j < len; j++) {
            if (!sym[j] || sym[j] != str[j]) {
                match = false;
                break;
            }
        }
        if (match && !sym[len]) {
            return i;  // Found known symbol
        }
    }
    
    return 0x8000 | (hash & 0x7FFF);  // Unknown symbol
}

__device__ void parse_expression(
    const char* source,
    uint32_t start,
    uint32_t end,
    SExpression* expr
) {
    expr->token_count = 0;
    expr->depth = 0;
    
    uint32_t pos = start;
    while (pos < end && expr->token_count < MAX_EXPR_TOKENS) {
        // Skip whitespace
        while (pos < end && (source[pos] == ' ' || source[pos] == '\n' || 
                            source[pos] == '\t' || source[pos] == '\r')) {
            pos++;
        }
        
        if (pos >= end) break;
        
        Token* tok = &expr->tokens[expr->token_count];
        tok->source_offset = pos - start;
        
        if (source[pos] == '(') {
            tok->type = TOK_LPAREN;
            tok->length = 1;
            expr->depth++;
            pos++;
        } else if (source[pos] == ')') {
            tok->type = TOK_RPAREN;
            tok->length = 1;
            expr->depth--;
            pos++;
        } else if (source[pos] == '"') {
            // String
            tok->type = TOK_STRING;
            uint32_t str_start = ++pos;
            while (pos < end && source[pos] != '"') pos++;
            tok->value.string_idx = str_start;
            tok->length = pos - str_start;
            if (pos < end) pos++; // Skip closing quote
        } else if ((source[pos] >= '0' && source[pos] <= '9') || 
                   source[pos] == '-' || source[pos] == '.') {
            // Number
            tok->type = TOK_NUMBER;
            uint32_t num_start = pos;
            bool is_float = false;
            if (source[pos] == '-') pos++;
            while (pos < end && ((source[pos] >= '0' && source[pos] <= '9') || 
                                source[pos] == '.')) {
                if (source[pos] == '.') is_float = true;
                pos++;
            }
            
            // Simple atof
            float val = 0.0f;
            float sign = (source[num_start] == '-') ? -1.0f : 1.0f;
            uint32_t i = (source[num_start] == '-') ? num_start + 1 : num_start;
            float decimal = 0.1f;
            bool after_decimal = false;
            
            while (i < pos) {
                if (source[i] == '.') {
                    after_decimal = true;
                } else {
                    float digit = source[i] - '0';
                    if (after_decimal) {
                        val += digit * decimal;
                        decimal *= 0.1f;
                    } else {
                        val = val * 10.0f + digit;
                    }
                }
                i++;
            }
            
            tok->value.number = val * sign;
            tok->length = pos - num_start;
        } else {
            // Symbol
            tok->type = TOK_SYMBOL;
            uint32_t sym_start = pos;
            while (pos < end && source[pos] != '(' && source[pos] != ')' && 
                   source[pos] != ' ' && source[pos] != '\n' && 
                   source[pos] != '\t' && source[pos] != '\r') {
                pos++;
            }
            tok->value.symbol_id = parse_symbol(&source[sym_start], pos - sym_start);
            tok->length = pos - sym_start;
        }
        
        expr->token_count++;
    }
}

// ============================================================================
// Parallel Expression Evaluation VM
// ============================================================================

struct VMValue {
    enum Type { NIL, NUMBER, SYMBOL, LIST, BUILTIN } type;
    union {
        float number;
        uint32_t symbol;
        uint32_t list_idx;
        uint32_t builtin_id;
    } data;
};

struct VMEnv {
    VMValue stack[256];
    uint32_t sp;
    
    // Output commands
    uint32_t* output_buffer;
    uint32_t output_offset;
    uint32_t max_output;
};

__device__ VMValue eval_expr(SExpression* expr, uint32_t start_token, 
                             uint32_t end_token, VMEnv* env);

__device__ VMValue eval_list(SExpression* expr, uint32_t start_token, 
                             uint32_t end_token, VMEnv* env) {
    if (start_token >= end_token) {
        return {VMValue::NIL, {0}};
    }
    
    Token* first = &expr->tokens[start_token];
    
    // Check for special forms
    if (first->type == TOK_SYMBOL) {
        uint32_t sym = first->value.symbol_id;
        
        // Arithmetic operators
        if (sym == 15) {  // "+"
            VMValue result = {VMValue::NUMBER, {0.0f}};
            for (uint32_t i = start_token + 1; i < end_token; i++) {
                VMValue arg = eval_expr(expr, i, i + 1, env);
                if (arg.type == VMValue::NUMBER) {
                    result.data.number += arg.data.number;
                }
            }
            return result;
        } else if (sym == 16) {  // "-"
            if (start_token + 1 < end_token) {
                VMValue first = eval_expr(expr, start_token + 1, start_token + 2, env);
                if (start_token + 2 >= end_token) {
                    // Unary minus
                    first.data.number = -first.data.number;
                    return first;
                }
                // Binary minus
                VMValue second = eval_expr(expr, start_token + 2, start_token + 3, env);
                first.data.number -= second.data.number;
                return first;
            }
        } else if (sym == 17) {  // "*"
            VMValue result = {VMValue::NUMBER, {1.0f}};
            for (uint32_t i = start_token + 1; i < end_token; i++) {
                VMValue arg = eval_expr(expr, i, i + 1, env);
                if (arg.type == VMValue::NUMBER) {
                    result.data.number *= arg.data.number;
                }
            }
            return result;
        }
        // Graphics commands
        else if (sym == 6) {  // "draw"
            // Emit draw command
            if (env->output_offset + 5 < env->max_output) {
                env->output_buffer[env->output_offset++] = 0x1000;  // DRAW_CMD
                
                // Get arguments
                for (uint32_t i = start_token + 1; i < end_token && i < start_token + 5; i++) {
                    VMValue arg = eval_expr(expr, i, i + 1, env);
                    env->output_buffer[env->output_offset++] = 
                        (arg.type == VMValue::NUMBER) ? (uint32_t)arg.data.number : 0;
                }
            }
            return {VMValue::NIL, {0}};
        } else if (sym == 9) {  // "rect"
            // Emit rectangle command
            if (env->output_offset + 5 < env->max_output) {
                env->output_buffer[env->output_offset++] = 0x1001;  // RECT_CMD
                
                // Get x, y, width, height
                for (uint32_t i = start_token + 1; i < end_token && i < start_token + 5; i++) {
                    VMValue arg = eval_expr(expr, i, i + 1, env);
                    env->output_buffer[env->output_offset++] = 
                        (arg.type == VMValue::NUMBER) ? (uint32_t)arg.data.number : 0;
                }
            }
            return {VMValue::NIL, {0}};
        }
    }
    
    return {VMValue::NIL, {0}};
}

__device__ VMValue eval_expr(SExpression* expr, uint32_t start_token, 
                             uint32_t end_token, VMEnv* env) {
    if (start_token >= end_token) {
        return {VMValue::NIL, {0}};
    }
    
    Token* tok = &expr->tokens[start_token];
    
    switch (tok->type) {
        case TOK_NUMBER:
            return {VMValue::NUMBER, {tok->value.number}};
            
        case TOK_SYMBOL:
            return {VMValue::SYMBOL, {tok->value.symbol_id}};
            
        case TOK_LPAREN: {
            // Find matching rparen
            uint32_t depth = 1;
            uint32_t end = start_token + 1;
            while (end < end_token && depth > 0) {
                if (expr->tokens[end].type == TOK_LPAREN) depth++;
                else if (expr->tokens[end].type == TOK_RPAREN) depth--;
                end++;
            }
            return eval_list(expr, start_token + 1, end - 1, env);
        }
        
        default:
            return {VMValue::NIL, {0}};
    }
}

// ============================================================================
// Main Parallel Lisp Kernel
// ============================================================================

__global__ void gpu_lisp_eval_kernel(
    const char* source,
    uint32_t source_len,
    uint32_t* output_commands,
    uint32_t max_commands_per_thread
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for expression boundaries
    __shared__ uint32_t expr_starts[1024];
    __shared__ uint32_t expr_ends[1024];
    __shared__ uint32_t expr_count;
    
    if (threadIdx.x == 0) {
        expr_count = 0;
    }
    __syncthreads();
    
    // Phase 1: Find all expressions in parallel
    find_expression_boundaries(source, source_len, expr_starts, expr_ends, 
                               &expr_count, threadIdx.x, blockDim.x);
    __syncthreads();
    
    // Phase 2: Each thread parses and evaluates one expression
    if (tid < expr_count) {
        SExpression expr;
        parse_expression(source, expr_starts[tid], expr_ends[tid], &expr);
        
        // Setup VM environment
        VMEnv env;
        env.sp = 0;
        env.output_buffer = &output_commands[tid * max_commands_per_thread];
        env.output_offset = 0;
        env.max_output = max_commands_per_thread;
        
        // Evaluate expression
        eval_expr(&expr, 0, expr.token_count, &env);
        
        // Mark end of commands
        if (env.output_offset < env.max_output) {
            env.output_buffer[env.output_offset] = 0;  // Sentinel
        }
    }
}

// ============================================================================
// Test Interface
// ============================================================================

extern "C" void test_gpu_lisp() {
    // Test Lisp program
    const char* program = 
        "(draw 100 100 50 50)\n"
        "(rect 200 200 (* 10 10) (+ 40 20))\n"
        "(draw (+ 1 2 3) (* 5 5) 100 100)\n"
        "(rect 0 0 1920 1080)\n";
    
    uint32_t len = strlen(program);
    
    // Allocate GPU memory
    char* d_source;
    uint32_t* d_output;
    cudaMalloc(&d_source, len + 1);
    cudaMalloc(&d_output, 1024 * 100 * sizeof(uint32_t));  // 100 commands per expr
    cudaMemset(d_output, 0, 1024 * 100 * sizeof(uint32_t));
    
    // Copy source to GPU
    cudaMemcpy(d_source, program, len + 1, cudaMemcpyHostToDevice);
    
    // Launch kernel
    printf("Launching GPU Lisp evaluator...\n");
    gpu_lisp_eval_kernel<<<1, 256>>>(d_source, len, d_output, 100);
    cudaDeviceSynchronize();
    
    // Read back results
    uint32_t h_output[1024 * 100];
    cudaMemcpy(h_output, d_output, 1024 * 100 * sizeof(uint32_t), 
               cudaMemcpyDeviceToHost);
    
    // Print generated commands
    printf("\nGenerated Commands:\n");
    for (int expr = 0; expr < 10; expr++) {
        uint32_t* cmds = &h_output[expr * 100];
        if (cmds[0] == 0) continue;
        
        printf("Expression %d:\n", expr);
        for (int i = 0; i < 100 && cmds[i] != 0; i++) {
            if (cmds[i] == 0x1000) {
                printf("  DRAW: x=%u y=%u w=%u h=%u\n", 
                       cmds[i+1], cmds[i+2], cmds[i+3], cmds[i+4]);
                i += 4;
            } else if (cmds[i] == 0x1001) {
                printf("  RECT: x=%u y=%u w=%u h=%u\n",
                       cmds[i+1], cmds[i+2], cmds[i+3], cmds[i+4]);
                i += 4;
            }
        }
    }
    
    cudaFree(d_source);
    cudaFree(d_output);
}

int main() {
    test_gpu_lisp();
    return 0;
}
