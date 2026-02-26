// include/config.h
#ifndef CONFIG_H
#define CONFIG_H

// Define common constants
#define DEFAULT_ARRAY_SIZE 1024
#define BLOCK_SIZE 256

// You could also define simple configuration flags
#define ENABLE_DEBUG_MESSAGES 1
#define INPUTgRAPH "/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-11/holl11_csr"
#define INPUTpmAT "/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-11/holl11_csr_P"
#define REPEAT_MULT 1000
#define IS_WEIGHTED 0 // 1 if grpah is weighted
#define RESIDUAL_RUN 0
#define RESIDUAL_THRESHOLD 1000000
#define GPU_COUNT 1
#define SOLVER_COUNT 1
#define IS_SOLVER 1 // 0: for betweeness centrality calculation
#define OpFileSolK "/home/sumaiya/FixedPS/kSolve/data/outputs/holl11/holl11_k131072"
#define OpFileSolM "/home/sumaiya/FixedPS/kSolve/data/outputs/holl11/holl11_m"
#define NO_OF_MSJ 131072
#define MAX_EPOCH 10
#endif // CONFIG_H
