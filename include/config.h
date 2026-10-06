// include/config.h
#ifndef CONFIG_H
#define CONFIG_H

// Define common constants
#define DEFAULT_ARRAY_SIZE 1024
#define BLOCK_SIZE 256

// Configuration flags
#define ENABLE_DEBUG_MESSAGES 1

// Linux POSIX file paths (forward slashes, no drive letters)
#define INPUTgRAPH "/home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/as_ski/as_ski_csr_b11-sink"
#define INPUTpmAT "/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/as-ski/as-skitter.txt_csr_P"

#define REPEAT_MULT 1000
#define IS_WEIGHTED 0 // 1 if graph is weighted
#define RESIDUAL_RUN 0
#define RESIDUAL_THRESHOLD 1000000
#define GPU_COUNT 0
#define SOLVER_COUNT 1
#define IS_SOLVER 1 // 0: for betweenness centrality calculation

// Output paths updated for Linux
#define OpFileSolK "/home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/as_ski/asSki_b11_262144"
#define OpFileSolM "/home/sumaiya/FixedPS/kSolve/data/outputs/asSki/asSki_m"

#define NO_OF_MSJ 262144
#define MAX_EPOCH 15625

#endif // CONFIG_H
