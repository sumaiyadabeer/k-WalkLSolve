// include/modules/module1.cuh
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
#include <thrust/iterator/constant_iterator.h>
#ifdef __cplusplus
 extern "C" {

struct message{
    int msg_idx;
    int node;
};

int read_file_into_csr(const char* file_path, int** row_ptr, int** col_offset, float** values, float** b);


int read_csr(const char* file_path, int** row_ptr, int** col_ptr, float** values, float** b);


int b2J(int V, float* b, float* neg_value, int* neg_index, int &source_nodes, int &sink_nodes);


void print_csr(int V, int* row_ptr, int* col_offset, float* values);



__global__ void flush_boolArray(bool *d_recv, int V);

__global__ void setup_kernel(curandState *state, unsigned long seed); 

__device__ int get_source_node(curandState *state, int *d_source_idx, int source_nodes, int V);

__device__ int get_nbr(curandState *state, int* row_ptr, int* col_offset, float* values, int node); 

__global__ void processMessages(int *d_row_ptr, int *d_col_offset, float *d_values, struct message *d_messages,
                                int num_msg_packets, int V,float *d_b ,int *d_source_idx, int source_nodes, 
                                curandState *state, bool* d_recv, int neg_index, int *d_traveller); 



__global__ void update_eta_by_qLen(float* d_eta, float* d_prev_eta ,int *d_cnt, int V, bool *d_recv, int epoch, int num_msg_packets);

__global__ void normalise(float *d_eta, int V, int epoch);

__global__ void eta2x(float *d_eta, int V, int no_of_msjs, int *d_row_ptr);

__global__ void update_qlen(int *d_traveller, int num_msg_packets, int *d_cnt);

__global__ void cnt_diff(int *d_cnt, int *del_d_cnt, int V);
// // Declare the kernel here
// __global__ void addKernel(const float* a, const float* b, float* c, int size);

// cudaError_t addArrays(const float* d_a, const float* d_b, float* d_c, int size);
// bool runArrayAddition(const float* h_a, const float* h_b, float* h_c, int size);

}
#endif

#endif // MODULE1_CUH