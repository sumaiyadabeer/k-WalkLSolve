// #include/modules/helper.cuh
#include <cuda_runtime.h>
#include "../include/modules/SolverM.cuh" //for sparse matrix defination
#include "../include/config.h" //for sparse matrix defination
#include <iostream>
#include <vector>
// Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUSPARSE(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

bool readPMatrixFromCSRFile(const std::string& filename, int& n, 
    std::vector<int>& row_ptr, 
    std::vector<int>& col_idx, 
    std::vector<float>& values,
    std::vector<float>& start_dis,
    int& sink_index,
    float& sink_value);

bool readGMatrixFromCSRFile(const std::string& filename, int& n, 
        std::vector<int>& row_ptr, 
        std::vector<int>& col_idx, 
        std::vector<float>& values,
        std::vector<float>& b);

float Wrapper_calculateLaplacianError(int sink_index, int n, int e, float* sol,
            int* row_ptr, int* col_idx, float* b,
            bool use_cusparse);
float calculateLaplacianError(int sink_index, int n, std::vector<float>& sol,
                const std::vector<int>& row_ptr, const std::vector<int>& col_idx,
                const std::vector<float>& values, const std::vector<float>& b,
                bool use_cusparse);
std::vector<float> calculateAbsoluteDifferenceSum(int n, std::vector<float>& sol,
                    std::vector<int>& row_ptr,
                    std::vector<int>& col_idx);

__global__ void calculate_abs_diff_sum_kernel(int n,  int* row_ptr,
                        int* col_idx,  float* sol,
                       float* x_out);