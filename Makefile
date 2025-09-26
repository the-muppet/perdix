# Makefile for Perdix GPU System
# Refactored modular architecture

# Compiler and flags
NVCC = nvcc
CUDA_ARCH = -arch=sm_80  # Ampere and later (change for your GPU)
NVCC_FLAGS = -O3 -std=c++14 -Xcompiler -Wall -lineinfo
NVCC_DEBUG_FLAGS = -g -G -DDEBUG

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Source files
COMMON_HEADER = $(SRC_DIR)/perdix_common.cuh
MODULES = ring_buffer.cu text_processing.cu rendering.cu
MODULE_SRCS = $(addprefix $(SRC_DIR)/, $(MODULES))
MODULE_OBJS = $(addprefix $(BUILD_DIR)/, $(MODULES:.cu=.o))

# Test programs
TEST_RING = test_ring_buffer
TEST_TEXT = test_text_processing
TEST_RENDER = test_rendering

# Default target
all: dirs $(MODULE_OBJS)
	@echo "✓ All modules compiled successfully"

# Create directories
dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Compile modules
$(BUILD_DIR)/ring_buffer.o: $(SRC_DIR)/ring_buffer.cu $(COMMON_HEADER)
	@echo "Compiling ring_buffer.cu..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -c $< -o $@

$(BUILD_DIR)/text_processing.o: $(SRC_DIR)/text_processing.cu $(COMMON_HEADER)
	@echo "Compiling text_processing.cu..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -c $< -o $@

$(BUILD_DIR)/rendering.o: $(SRC_DIR)/rendering.cu $(COMMON_HEADER)
	@echo "Compiling rendering.cu..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -c $< -o $@

# Build test programs
test_ring: $(BUILD_DIR)/ring_buffer.o
	@echo "Building ring buffer test..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -DTEST_RING_BUFFER \
		$(BUILD_DIR)/ring_buffer.o -o $(BIN_DIR)/$(TEST_RING)
	@echo "✓ Test program: $(BIN_DIR)/$(TEST_RING)"

test_text: $(BUILD_DIR)/text_processing.o
	@echo "Building text processing test..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -DTEST_TEXT_PROCESSING \
		$(BUILD_DIR)/text_processing.o -o $(BIN_DIR)/$(TEST_TEXT)
	@echo "✓ Test program: $(BIN_DIR)/$(TEST_TEXT)"

test_render: $(BUILD_DIR)/rendering.o
	@echo "Building rendering test..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -DTEST_RENDERING \
		$(BUILD_DIR)/rendering.o -o $(BIN_DIR)/$(TEST_RENDER)
	@echo "✓ Test program: $(BIN_DIR)/$(TEST_RENDER)"

# Build all tests
tests: test_ring test_text test_render
	@echo "✓ All test programs built"

# Build complete system (example)
perdix: $(MODULE_OBJS)
	@echo "Linking Perdix system..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(MODULE_OBJS) \
		-o $(BIN_DIR)/perdix
	@echo "✓ Perdix system built: $(BIN_DIR)/perdix"

# Debug builds
debug: NVCC_FLAGS = $(NVCC_DEBUG_FLAGS)
debug: all

# Performance build with all optimizations
perf: NVCC_FLAGS = -O3 -use_fast_math -Xptxas -dlcm=ca -lineinfo
perf: all

# Profile build for nvprof/nsight
profile: NVCC_FLAGS = -O3 -lineinfo -Xcompiler -rdynamic
profile: all

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✓ Clean complete"

# Run tests
run_tests: tests
	@echo "Running ring buffer test..."
	@cd $(BIN_DIR) && ./$(TEST_RING)
	@echo "\nRunning text processing test..."
	@cd $(BIN_DIR) && ./$(TEST_TEXT)
	@echo "\nRunning rendering test..."
	@cd $(BIN_DIR) && ./$(TEST_RENDER)

# Check GPU info
gpu_info:
	@echo "CUDA Device Information:"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
	@echo "\nCUDA Compiler Version:"
	@$(NVCC) --version

# Benchmark
benchmark: perf
	@echo "Running performance benchmarks..."
	@cd $(BIN_DIR) && ./perdix --benchmark

# Static analysis with cuda-memcheck
memcheck: debug
	@echo "Running CUDA memory checker..."
	@cuda-memcheck --leak-check full $(BIN_DIR)/perdix

# Generate PTX for inspection
ptx: $(MODULE_SRCS)
	@echo "Generating PTX files..."
	@mkdir -p $(BUILD_DIR)/ptx
	@for src in $(MODULE_SRCS); do \
		base=$$(basename $$src .cu); \
		$(NVCC) $(CUDA_ARCH) -ptx $$src -o $(BUILD_DIR)/ptx/$$base.ptx; \
		echo "  Generated $(BUILD_DIR)/ptx/$$base.ptx"; \
	done

# Install (copy to system location)
install: perdix
	@echo "Installing Perdix..."
	@sudo cp $(BIN_DIR)/perdix /usr/local/bin/
	@sudo cp $(COMMON_HEADER) /usr/local/include/
	@echo "✓ Installed to /usr/local/bin/perdix"

# Help
help:
	@echo "Perdix GPU System - Makefile Targets"
	@echo "====================================="
	@echo "  all         - Build all modules (default)"
	@echo "  perdix      - Build complete system"
	@echo "  tests       - Build all test programs"
	@echo "  run_tests   - Build and run all tests"
	@echo "  debug       - Build with debug symbols"
	@echo "  perf        - Build with maximum optimizations"
	@echo "  profile     - Build for profiling"
	@echo "  clean       - Remove all build artifacts"
	@echo "  gpu_info    - Display GPU information"
	@echo "  benchmark   - Run performance benchmarks"
	@echo "  memcheck    - Check for memory errors"
	@echo "  ptx         - Generate PTX assembly"
	@echo "  install     - Install to system"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH   - Target GPU architecture (default: sm_80)"
	@echo "  NVCC        - CUDA compiler (default: nvcc)"
	@echo ""
	@echo "Examples:"
	@echo "  make CUDA_ARCH=-arch=sm_75   # Build for Turing"
	@echo "  make debug run_tests          # Debug build and test"
	@echo "  make perf benchmark           # Performance benchmark"

.PHONY: all dirs clean tests run_tests debug perf profile \
        gpu_info benchmark memcheck ptx install help \
        test_ring test_text test_render perdix
