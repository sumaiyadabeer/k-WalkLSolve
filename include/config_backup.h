// include/config.h
#ifndef CONFIG_H
#define CONFIG_H

// Define common constants
#define DEFAULT_ARRAY_SIZE 1024
#define BLOCK_SIZE 256

// You could also define simple configuration flags
#define ENABLE_DEBUG_MESSAGES 1
#define INPUTgRAPH "/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikibug/bug_csr" //"/home/sumaiya/FixedPS/kSolve/data/inputs/matrix_input.txt" //
#define INPUTpmAT "/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikibug/bug_csr_P"//"/home/sumaiya/FixedPS/kSolve/data/inputs/matrix_input_P.txt"//
#define REPEAT_MULT 1000000
#define IS_WEIGHTED 0 // 1 if grpah is weighted
#define TRACKeRROR 0  // 1 for generation of log f error
#define ERRORpERiTER 10  // 1 for generation of log f error
#define GPU_COUNT 2
#define SOLVER_COUNT 0 // 0:both 1:ksolver 2:MultSolver
#define IS_SOLVER 1 // 0: for betweeness centrality calculation
#define OpFileSolK  "/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug_new_k"//"/home/sumaiya/FixedPS/kSolve/data/outputs/temp_k"//
#define OpFileSolM  "/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug_new_m"//"/home/sumaiya/FixedPS/kSolve/data/outputs/temp_m" //
// #define OpFileErr  "/home/sumaiya/FixedPS/kSolve/data/inputs/matrix_input.txt_sol"
#define NO_OF_MSJ  4196
#define MAX_EPOCH  1000000
#endif // CONFIG_H
