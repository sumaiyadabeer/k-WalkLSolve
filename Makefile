# Makefile for CUDA C++ Project

# --- Compiler Settings ---
NVCC = nvcc
# Specify CUDA compute capability (e.g., sm_75 for Turing, sm_86 for Ampere)
# Adjust this to match your GPU's architecture.
CUDA_ARCH = -gencode arch=compute_70,code=sm_70
# Removed -rdc=true as kernels are called from within their own .cu files.
# This simplifies linkage and can resolve "named symbol not found" errors.
NVCC_FLAGS = -std=c++17 -Xcompiler -Wall -Xcompiler -Wextra $(CUDA_ARCH) -lcublas -lcusparse

# --- Directories ---
BIN_DIR = bin
SRC_DIR = src
MODULES_SRC_DIR = $(SRC_DIR)/modules
INCLUDE_DIR = include
MODULES_INCLUDE_DIR = $(INCLUDE_DIR)/modules
TESTS_DIR = tests
DATA_INPUT_DIR = data/inputs
DATA_OUTPUT_DIR = data/outputs

# --- Include Paths ---
# -I tells the compiler where to look for header files
INCLUDES = -I$(INCLUDE_DIR) -I$(MODULES_INCLUDE_DIR)

# --- Source Files ---
MAIN_SRC = $(SRC_DIR)/main.cu
MODULE_SRCS = $(MODULES_SRC_DIR)/SolverM.cu $(MODULES_SRC_DIR)/kwalk.cu $(MODULES_SRC_DIR)/helper.cu
# TEST_SRCS = $(TESTS_DIR)/test_module1.cu $(TESTS_DIR)/test_module2.cu

# --- Object Files ---
# All object files will be placed directly in BIN_DIR
MAIN_OBJ = $(BIN_DIR)/main.o
# Use patsubst to correctly map module source paths to bin object paths
MODULE_OBJS = $(patsubst $(MODULES_SRC_DIR)/%.cu, $(BIN_DIR)/%.o, $(MODULE_SRCS))
# Use patsubst to correctly map test source paths to bin object paths
# TEST_OBJS = $(patsubst $(TESTS_DIR)/%.cu, $(BIN_DIR)/%.o, $(TEST_SRCS))

# Removed DEVICE_LINK_OBJ as -rdc=true is no longer used.

# --- Executables ---
MAIN_APP = $(BIN_DIR)/main_app
# TEST_APP1 = $(BIN_DIR)/test_module1
# TEST_APP2 = $(BIN_DIR)/test_module2

# --- Default Target ---
.PHONY: all
all: $(MAIN_APP) #$(TEST_APP1) $(TEST_APP2)

# --- Build Rules ---

# Rule to create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Specific rule for compiling main.cu
# It explicitly states that bin/main.o depends on src/main.cu
$(MAIN_OBJ): $(MAIN_SRC) $(INCLUDE_DIR)/config.h $(MODULES_INCLUDE_DIR)/*.cuh | $(BIN_DIR)
	@echo "Compiling $< to $@"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Generic rule for compiling module .cu files
# This pattern correctly maps src/modules/X.cu to bin/X.o
$(BIN_DIR)/%.o: $(MODULES_SRC_DIR)/%.cu $(INCLUDE_DIR)/config.h $(MODULES_INCLUDE_DIR)/*.cuh | $(BIN_DIR)
	@echo "Compiling $< to $@"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Generic rule for compiling test .cu files
# This pattern correctly maps tests/X.cu to bin/X.o
$(BIN_DIR)/%.o: $(TESTS_DIR)/%.cu $(INCLUDE_DIR)/config.h $(MODULES_INCLUDE_DIR)/*.cuh | $(BIN_DIR)
	@echo "Compiling $< to $@"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Removed explicit device link step rule.

# Build main application
# No longer depends on device_link.o
$(MAIN_APP): $(MAIN_OBJ) $(MODULE_OBJS) | $(BIN_DIR)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) $(MAIN_OBJ) $(MODULE_OBJS) -o $@

# Build test executable 1
$(TEST_APP1): $(BIN_DIR)/test_module1.o $(MODULE_OBJS) | $(BIN_DIR)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) $(BIN_DIR)/test_module1.o $(MODULE_OBJS) -o $@

# Build test executable 2
$(TEST_APP2): $(BIN_DIR)/test_module2.o $(MODULE_OBJS) | $(BIN_DIR)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) $(BIN_DIR)/test_module2.o $(MODULE_OBJS) -o $@

# --- Clean Rule ---
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(BIN_DIR)/*.o $(BIN_DIR)/* 

# --- Test Execution Rules (Optional, for convenience) ---
.PHONY: run_main
run_main: $(MAIN_APP)
	@echo "Running main application:"
	./$(MAIN_APP)

.PHONY: run_test1
run_test1: $(TEST_APP1)
	@echo "Running Module 1 tests:"
	./$(TEST_APP1)

.PHONY: run_test2
run_test2: $(TEST_APP2)
	@echo "Running Module 2 tests:"
	./$(TEST_APP2)

.PHONY: run_tests
run_tests: run_test1 run_test2
	@echo "All tests finished."
