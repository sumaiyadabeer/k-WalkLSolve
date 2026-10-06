# ==============================================================================
# Linux Makefile for CUDA C++ Project
# ==============================================================================

# Compiler & Path
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(shell command -v nvcc 2>/dev/null || echo $(CUDA_PATH)/bin/nvcc)

# GPU Architecture
CUDA_ARCH = -gencode arch=compute_80,code=sm_80

# Linux Settings
OBJ_EXT = o
EXE_EXT =
HOST_WARNINGS = -Xcompiler -Wall -Xcompiler -Wextra
MKDIR = mkdir -p $(BIN_DIR)
RM = rm -rf $(BIN_DIR)

# Directories
BIN_DIR = bin
SRC_DIR = src
MODULES_SRC_DIR = $(SRC_DIR)/modules
INCLUDE_DIR = include
MODULES_INCLUDE_DIR = $(INCLUDE_DIR)/modules

# Include Paths & Flags
INCLUDES = -I$(INCLUDE_DIR) -I$(MODULES_INCLUDE_DIR)
NVCC_FLAGS = -std=c++17 $(HOST_WARNINGS) $(CUDA_ARCH) -lcublas -lcusparse

# Source Files
MAIN_SRC = $(SRC_DIR)/main.cu
MODULE_SRCS = $(MODULES_SRC_DIR)/SolverM.cu $(MODULES_SRC_DIR)/kwalk.cu $(MODULES_SRC_DIR)/helper.cu

# Object Files & Executable
MAIN_OBJ = $(BIN_DIR)/main.$(OBJ_EXT)
MODULE_OBJS = $(patsubst $(MODULES_SRC_DIR)/%.cu, $(BIN_DIR)/%.$(OBJ_EXT), $(MODULE_SRCS))
MAIN_APP = $(BIN_DIR)/main_app$(EXE_EXT)

.PHONY: all clean run_main

all: $(MAIN_APP)

$(BIN_DIR):
	@$(MKDIR)

$(MAIN_OBJ): $(MAIN_SRC) $(INCLUDE_DIR)/config.h | $(BIN_DIR)
	@echo Compiling $< to $@
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%.$(OBJ_EXT): $(MODULES_SRC_DIR)/%.cu $(INCLUDE_DIR)/config.h | $(BIN_DIR)
	@echo Compiling $< to $@
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(MAIN_APP): $(MAIN_OBJ) $(MODULE_OBJS) | $(BIN_DIR)
	@echo Linking $@...
	$(NVCC) $(NVCC_FLAGS) $(MAIN_OBJ) $(MODULE_OBJS) -o $@

clean:
	@echo Cleaning up...
	@$(RM)

run_main: $(MAIN_APP)
	@echo Running main application:
	./$(MAIN_APP)