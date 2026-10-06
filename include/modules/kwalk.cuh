// src/modules/kwalk.cu
#ifndef MODULE2_CUH
#define MODULE2_CUH
#include <curand_kernel.h>
#include <stdio.h>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <iostream>

#ifdef __cplusplus
extern "C" {

struct message{
    int msg_idx;
    int node;
};

// Reads Graph Structure in CSR format
int read_csr(const char* file_path, int** row_ptr, int** col_ptr, float** values, float** b);

int b2J(int V, float* b, float* neg_value, int* neg_index, int &source_nodes, int &sink_nodes);
// Device-Side Philox Random Neighbor Selection
__device__ int get_nbr_philox(unsigned long long seed, int idx, int epoch,
                             const int* __restrict__ d_row_ptr,
                             const int* __restrict__ d_col_offset,
                             int curr_node);
                             


// Optimized Fused Message Processing Kernel
__global__ void processMessages_Optimized(
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_offset,
    const float* __restrict__ d_values,
    struct message* d_messages,
    int num_msg_packets,
    int V,
    const float* __restrict__ d_b,
    const int* __restrict__ d_source_idx,
    int source_nodes,
    int* d_traveller,
    int epoch,
    long long max_epoch,
    unsigned long long* d_at_sink,
    unsigned long long* d_walk_generated,
    int* d_cnt,
    unsigned long long seed);
// Active Walk Compaction (AWC) via Kogge-Stone Parallel Scan
__global__ void partitionValueToRight2(
    int* __restrict__ d_input, 
    int* __restrict__ d_output, 
    struct message* __restrict__ d_msg_in,
    struct message* __restrict__ d_msg_out,
    int n, 
    int s, 
    int* d_non_s_cnt,
    int* d_s_cnt);
    
// Single-Block Kernel for Top-R Extraction
__global__ void extractTopRKernel(const int* d_cnt, int V, int r, int* d_top_indices, int* d_top_values);

// Post-Processing Utility Kernels
__global__ void update_eta_by_q(float* d_eta, const int* d_cnt, int V, unsigned long long total_walks);
__global__ void eta2x(float *d_eta, int V, const int *d_row_ptr);

__global__ void flush_boolArray(bool *d_recv, int V);

__global__ void setup_kernel(curandState *state, unsigned long seed);
}
#endif

#endif // MODULE1_CUH