// src/modules/module1.cu
#include "../../include/modules/kwalk.cuh" // Relative path to its header
#include "../../include/config.h"           // Access global config
#include <curand_kernel.h>
#include <stdio.h>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/iterator/constant_iterator.h>


int read_file_into_csr(const char* file_path, int** row_ptr, int** col_offset, float** values, float** b) {
    FILE* graph = fopen(file_path, "r");
    if (graph == NULL) {
        fprintf(stderr, "Unable to open the file at %s\n", file_path);
        return -1;
    }

    int V;
    fscanf(graph, "%d", &V);
    int non_zero_values = 0;

    float degree[V];
    memset(degree, 0, sizeof(degree));

    float** matrix = (float**)malloc(sizeof(float*) * V);
    for (int i = 0; i < V; i++) {
        matrix[i] = (float*)malloc(sizeof(float) * V);
        memset(matrix[i], 0, sizeof(float) * V);  // Initialize the matrix to zero
    }

    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            fscanf(graph, "%f", &matrix[i][j]);
            if (matrix[i][j] != 0) {
                non_zero_values++;
                degree[i] += matrix[i][j];
            }
        }
    }

    // Normalize the weights
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (matrix[i][j] > 0) {
                matrix[i][j] = matrix[i][j] / degree[i];
            }
        }
    }

    *row_ptr = (int*)malloc(sizeof(int) * (V + 1));
    *col_offset = (int*)malloc(sizeof(int) * non_zero_values);
    *values = (float*)malloc(sizeof(float) * non_zero_values);
    *b = (float*)malloc(sizeof(float) * V);

    int r = 0;
    int c = 0;
    int v = 0;

    for (int i = 0; i < V; i++) {
        (*row_ptr)[r] = c;
        for (int j = 0; j < V; j++) {
            if (matrix[i][j] != 0) {
                (*col_offset)[c] = j;
                (*values)[v] = matrix[i][j];
                c++;
                v++;
            }
        }
        r++;
    }
    (*row_ptr)[r] = c;
    float sum = 0;
    for (int i = 0; i < V; i++) {
        fscanf(graph, "%f", &((*b)[i]));
        sum += (*b)[i];

    }


    for (int i = 0; i < V; i++) {
        free(matrix[i]);
    }
    free(matrix);

    fclose(graph);

    return V;
}


int read_csr(const char* file_path, int** row_ptr, int** col_ptr, float** values, float** b) {
    // printf(".\n");

    FILE *file = fopen(file_path, "r");
    if (file == nullptr) {
        perror("Failed to open file");
        return -1; 
    }
    // printf(".\n");


    int V, E;
    if (fscanf(file, "%d", &V) != 1 || fscanf(file, "%d", &E) != 1) {
        perror("Failed to read V or E");
        fclose(file);
        return -1; 
    }
    // printf(".\n");


    *row_ptr = (int*)malloc(sizeof(int) * (V + 1));
    *col_ptr = (int*)malloc(sizeof(int) * E);
    *values = (float*)malloc(sizeof(int) * E);
    *b = (float*)malloc(sizeof(float) * V);

    if (*row_ptr == nullptr || *col_ptr == nullptr || *values == nullptr || *b == nullptr) {
        perror("Failed to allocate memory");
        fclose(file);
        return -1; 
    }
    // printf(".\n");

    for (int i = 0; i <= V; ++i) {
        if (fscanf(file, "%d", &(*row_ptr)[i]) != 1) {
            perror("Failed to read row_ptr");
            fclose(file);
            return -1; 
        }
    }

    // printf(".\n");

    for (int i = 0; i < E; ++i) {
        if (fscanf(file, "%d", &(*col_ptr)[i]) != 1) {
            perror("Failed to read col_ptr");
            fclose(file);
            return -1; 
        }
    }

    // printf(".\n");

    for (int i = 0; i < E; ++i) {
        if (fscanf(file, "%f", &(*values)[i]) != 1) {
            perror("Failed to read values");
            fclose(file);
            return -1; 
        }
    }

    // printf("Reading b starts.\n");

    for (int i = 0; i < V; ++i) {
        if (fscanf(file, "%f", &(*b)[i]) != 1) {
            perror("Failed to read b");
            fclose(file);
            return -1; 
        }
    }
    // printf("reading b from file end but file not closed\n");

    for(int i = 0 ; i<V ; i++){
        int start_idx = (*row_ptr)[i];
        int end_idx = (*row_ptr)[i+1];
        float sum = 0;
        for(int j = start_idx; j< end_idx ; j++){
            sum += (*values)[j];
        }
        // printf("start_idx, end_idx, sum: %d, %d, %f\n", start_idx, end_idx, sum);
        for(int j = start_idx; j< end_idx ; j++){
            (*values)[j] =  (*values)[j]/sum;
        }
    }

    fclose(file);
    // printf("file read done");
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

void print_csr(int V, int* row_ptr, int* col_offset, float* values) {
    for (int i = 0; i <= V; i++) {
        printf("%d ", row_ptr[i]);
    }
    printf("\n");
    for (int i = 0; i < V; i++) {
        int curr = row_ptr[i];
        int end = row_ptr[i + 1];
        if (end - 1 < curr) continue;
        for (int j = curr; j < end; j++) {
            printf("%d ", col_offset[j]);
        }
    }
    printf("\n");

    for (int i = 0; i < V; i++) {
        int curr = row_ptr[i];
        int end = row_ptr[i + 1];
        if (end - 1 < curr) continue;
        for (int j = curr; j < end; j++) {
            printf("%f ", values[j]);
        }
    }
    printf("\n");
    
    
}


__global__ void flush_boolArray(bool *d_recv, int V){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= V) return;
    d_recv[idx] = 0;

}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ int get_source_node(curandState *state, int *d_source_idx, int source_nodes, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    float random_number = curand_uniform(&localState);  // Generate random float in [0, 1]
    state[idx] = localState;  // Store the state back
    int lower_bound = 0;
    int upper_bound = source_nodes;
    int node_idx = lower_bound + (int)(random_number * (upper_bound - lower_bound));
    int node = d_source_idx[node_idx];
    return node;
}

__device__ int get_nbr(curandState *state, int* row_ptr, int* col_offset, float* values, int node) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    float random_number = curand_uniform(&localState);  // Generate random float in [0, 1]
    state[idx] = localState;  // Store the state back
    int lower_bound = row_ptr[node];
    int upper_bound = row_ptr[node+1];
    int nbr_node_idx = lower_bound + (int)(random_number * (upper_bound - lower_bound));
    int nbr_node = col_offset[nbr_node_idx];

    return nbr_node;
}

__global__ void processMessages(int *d_row_ptr, int *d_col_offset, float *d_values, struct message *d_messages,
                                int num_msg_packets, int V, float *d_b ,int *d_source_idx, int source_nodes, 
                                curandState *state, bool* d_recv, int neg_index, int *d_traveller) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // print("hy")
    if (idx < num_msg_packets) {
        struct message *msg = &d_messages[idx];
        int curr_node = msg->node;

        if (d_b[curr_node] < 0) {
            int src_node = get_source_node(state, d_source_idx, source_nodes,V);  // Device-compatible random number generation
            msg->node = src_node;
            d_traveller[idx] = src_node;
            d_recv[src_node] = 1;
        } else {
            int nbr = get_nbr(state, d_row_ptr, d_col_offset, d_values, curr_node);  // Device-compatible random number generation
            msg->node = nbr;
            
            d_recv[nbr] = 1;
            d_traveller[idx] = nbr;
        }
    }
}


__global__ void update_eta_by_qLen(float* d_eta, float* d_prev_eta ,int *d_cnt, int V, bool *d_recv, int epoch, int num_msg_packets) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V) return;
    d_prev_eta[idx] = d_eta[idx];
    d_eta[idx] = ((epoch-1)*d_eta[idx] + (1*d_cnt[idx]))/epoch;
    d_recv[idx] = 0;
    d_cnt[idx] = 0;

}

__global__ void normalise(float *d_eta, int V, int epoch){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V) return;
    d_eta[idx] = d_eta[idx]/epoch;
}

__global__ void eta2x(float *d_eta, int V, int no_of_msjs, int *d_row_ptr){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V) return;
    d_eta[idx] = d_eta[idx]/((d_row_ptr[idx+1]-d_row_ptr[idx])*no_of_msjs);
}

__global__ void cnt_diff(int *d_cnt, int *del_d_cnt, int V){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V) return;
    del_d_cnt[idx] = del_d_cnt[idx] - d_cnt[idx];
}

__global__ void update_qlen(int *d_traveller, int num_msg_packets, int *d_cnt){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num_msg_packets) return;

    int node = d_traveller[idx];
    atomicAdd(&d_cnt[node], 1);
}




