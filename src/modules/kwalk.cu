// src/modules/kwalk.cu
#include "../../include/modules/kwalk.cuh"
#include "../../include/config.h"
#include <curand_kernel.h>
#include <stdio.h>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <iostream>

// Reads Graph Structure in CSR format
int read_csr(const char* file_path, int** row_ptr, int** col_ptr, float** values, float** b) {
    FILE *file = fopen(file_path, "r");
    if (file == nullptr) {
        perror("Failed to open graph file");
        return -1; 
    }

    int V, E;
    if (fscanf(file, "%d", &V) != 1 || fscanf(file, "%d", &E) != 1) {
        perror("Failed to read V or E dimensions");
        fclose(file);
        return -1; 
    }

    *row_ptr = (int*)malloc(sizeof(int) * (V + 1));
    *col_ptr = (int*)malloc(sizeof(int) * E);
    *values  = (float*)malloc(sizeof(float) * E);
    *b       = (float*)malloc(sizeof(float) * V);

    if (*row_ptr == nullptr || *col_ptr == nullptr || *values == nullptr || *b == nullptr) {
        perror("Failed host memory allocation");
        fclose(file);
        return -1; 
    }

    for (int i = 0; i <= V; ++i) fscanf(file, "%d", &(*row_ptr)[i]);
    for (int i = 0; i < E; ++i)  fscanf(file, "%d", &(*col_ptr)[i]);
    for (int i = 0; i < E; ++i)  fscanf(file, "%f", &(*values)[i]);
    for (int i = 0; i < V; ++i)  fscanf(file, "%f", &(*b)[i]);

    // Row degree normalization
    for (int i = 0; i < V; i++) {
        int start_idx = (*row_ptr)[i];
        int end_idx   = (*row_ptr)[i + 1];
        float sum = 0.0f;
        for (int j = start_idx; j < end_idx; j++) {
            sum += (*values)[j];
        }
        if (sum > 0.0f) {
            for (int j = start_idx; j < end_idx; j++) {
                (*values)[j] = (*values)[j] / sum;
            }
        }
    }

    fclose(file);
    return V;
}

int b2J(int V, float* b, float* neg_value, int* neg_index, int &source_nodes, int &sink_nodes) {
    *neg_value = 0;
    for (int i = 0; i < V; i++) {
        if (b[i] < 0) {
            *neg_value = b[i];
            *neg_index = i;
            sink_nodes++;
        }
    }
    *neg_value = -(*neg_value);
    for (int i = 0; i < V; i++) {
        if (b[i] > 0) {
            b[i] = b[i] / *neg_value;
            source_nodes++;
        }
    }
    return 0;
}

// Device-Side Philox Random Neighbor Selection
__device__ int get_nbr_philox(unsigned long long seed, int idx, int epoch,
                             const int* __restrict__ d_row_ptr,
                             const int* __restrict__ d_col_offset,
                             int curr_node) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, epoch, &state);
    
    int start = __ldg(&d_row_ptr[curr_node]);
    int end   = __ldg(&d_row_ptr[curr_node + 1]);
    int degree = end - start;
    if (degree <= 0) return curr_node;

    int offset = (int)(curand_uniform(&state) * degree);
    if (offset >= degree) offset = degree - 1;
    
    return __ldg(&d_col_offset[start + offset]);
}

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
    unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_msg_packets) return;

    int curr_node = d_messages[idx].node;
    if (curr_node < 0) return; // Inactive walker guard

    float b_val = __ldg(&d_b[curr_node]);
    int next_node;
    if (b_val < 0) { // Current node is a Sink Vertex
        // Count the arrival step at the sink vertex
        atomicAdd(&d_cnt[curr_node], 1);

        if (epoch < max_epoch) {
            // Look-ahead / Restart Phase: Respawn walker at a random source node
            curandStatePhilox4_32_10_t state;
            curand_init(seed, idx, epoch, &state);
            int src_idx = (int)(curand_uniform(&state) * source_nodes);
            next_node = __ldg(&d_source_idx[src_idx]);
            
            atomicAdd(d_walk_generated, 1ULL);
            atomicAdd(&d_cnt[next_node], 1); // Record initial visit at respawned source
        } else {
            // Final Phase: Terminate walker cleanly and increment sink counter EXACTLY ONCE
            atomicAdd(d_at_sink, 1ULL);
            d_messages[idx].node = -1;
            d_traveller[idx] = -1;
            return;
        }
    } else { // Regular Transition
        next_node = get_nbr_philox(seed, idx, epoch, d_row_ptr, d_col_offset, curr_node);
        atomicAdd(&d_cnt[next_node], 1);
    }

    // Update State Arrays
    d_messages[idx].node = next_node;
    d_traveller[idx] = next_node;
}

// Active Walk Compaction (AWC) via Kogge-Stone Parallel Scan
__global__ void partitionValueToRight2(
    int* __restrict__ d_input, 
    int* __restrict__ d_output, 
    struct message* __restrict__ d_msg_in,
    struct message* __restrict__ d_msg_out,
    int n, 
    int s, 
    int* d_non_s_cnt,
    int* d_s_cnt) {
    
    extern __shared__ int temp_scan[]; 

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int is_non_s = (idx < n && d_input[idx] != s) ? 1 : 0;
    int is_s     = (idx < n && d_input[idx] == s) ? 1 : 0;
    
    temp_scan[tid] = is_non_s;
    __syncthreads();

    // Kogge-Stone Shared Memory Scan
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int val = (tid >= stride) ? temp_scan[tid - stride] : 0;
        __syncthreads();
        if (tid >= stride) temp_scan[tid] += val;
        __syncthreads();
    }

    int block_exclusive_non_s = (tid == 0) ? 0 : temp_scan[tid - 1];
    
    int valid_threads_in_block = min(n - (int)(blockIdx.x * blockDim.x), (int)blockDim.x);
    if (valid_threads_in_block < 0) valid_threads_in_block = 0;

    int block_total_non_s = (valid_threads_in_block > 0) ? temp_scan[valid_threads_in_block - 1] : 0;
    int block_total_s     = (valid_threads_in_block > 0) ? (valid_threads_in_block - block_total_non_s) : 0;
    __syncthreads();

    __shared__ int global_base_non_s;
    __shared__ int global_base_s;

    if (tid == 0) {
        global_base_non_s = atomicAdd(d_non_s_cnt, block_total_non_s);
        global_base_s     = atomicAdd(d_s_cnt, block_total_s);
    }
    __syncthreads();

    if (idx < n) {
        if (is_non_s) {
            int dest_idx = global_base_non_s + block_exclusive_non_s;
            d_output[dest_idx] = d_input[idx];
            d_msg_out[dest_idx] = d_msg_in[idx];
        } else if (is_s) {
            int block_exclusive_s = tid - block_exclusive_non_s;
            int tail_idx = n - 1 - (global_base_s + block_exclusive_s);
            
            if (tail_idx >= 0 && tail_idx < n) {
                d_output[tail_idx] = s;
                d_msg_out[tail_idx] = d_msg_in[idx];
            }
        }
    }
}

// Single-Block Kernel for Top-R Extraction
__global__ void extractTopRKernel(const int* d_cnt, int V, int r, int* d_top_indices, int* d_top_values) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < r; i++) {
            d_top_values[i] = -1;
            d_top_indices[i] = -1;
        }

        for (int node = 0; node < V; node++) {
            int count = d_cnt[node];

            for (int i = 0; i < r; i++) {
                if (count > d_top_values[i]) {
                    for (int j = r - 1; j > i; j--) {
                        d_top_values[j] = d_top_values[j - 1];
                        d_top_indices[j] = d_top_indices[j - 1];
                    }
                    d_top_values[i] = count;
                    d_top_indices[i] = node;
                    break;
                }
            }
        }
    }
}

// Post-Processing Utility Kernels
__global__ void update_eta_by_q(float* d_eta, const int* d_cnt, int V, unsigned long long total_walks) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V) return;
    d_eta[idx] = static_cast<float>(d_cnt[idx]) / static_cast<float>(total_walks);
}

__global__ void eta2x(float *d_eta, int V, const int *d_row_ptr) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V) return;
    int degree = d_row_ptr[idx + 1] - d_row_ptr[idx];
    if (degree > 0) {
        d_eta[idx] = d_eta[idx] / degree;
    }
}

__global__ void flush_boolArray(bool *d_recv, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V) return;
    d_recv[idx] = false;
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}