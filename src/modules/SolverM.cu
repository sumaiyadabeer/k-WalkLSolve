// src/modules/module1.cu
#include "../../include/modules/SolverM.cuh" // Relative path to its header
#include "../../include/config.h"           // Access global config
#include <iostream>




// Custom CUDA kernel for SpMV (alternative implementation)
__global__ void spmv_csr_kernel(int n, const int* row_ptr, const int* col_idx, 
    const float* values, const float* x, float* y) {
int row = blockIdx.x * blockDim.x + threadIdx.x;

if (row < n) {
float sum = 0.0f;
int start = row_ptr[row];
int end = row_ptr[row + 1];

for (int j = start; j < end; j++) {
sum += values[j] * x[col_idx[j]];
}

y[row] = sum;
}
}

// Repeated multiplication of result vector t times (y = A * (A * (... (A * x)...)))
std::vector<float>  repeatedMatrixVectorMultiplication(SparseMatrix& matrix, 
    std::vector<float>& initial_vector, 
    std::vector<float>& degree, 
    int t_times, bool use_cusparse) {
int n = matrix.getSize();
std::vector<float> current_vector = initial_vector;
std::vector<float> result_vector(n);

std::cout << "\n--- Performing " << t_times << " repeated matrix-vector multiplications (" 
<< (use_cusparse ? "cuSPARSE" : "Custom Kernel") << ") ---" << std::endl;

auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < t_times; ++i) {
if (use_cusparse) {
matrix.multiply_cusparse(current_vector, result_vector);
} else {
matrix.multiply_custom(current_vector, result_vector);
}

if(i==t_times-1){
    float err = 0;
    for(int j  =  1 ; j < n ; j++){
        err = max(err, abs(result_vector[j] - current_vector[j])/current_vector[j]);
    }
    printf("Max eta_delta (MM)= %f\n", err);}
current_vector = result_vector; // The output of current multiplication becomes input for the next
// handling infinity values in result
    if((i%100)==0.0){
        // printf(".");
        // printf("Normalizing at %d\n: ", i);
        float sum = std::accumulate(current_vector.begin(), current_vector.end(), 0.0f);
        if (sum == 0.0f) {
            std::cout << "MM::Sum of vector elements is zero. Cannot divide by zero. Vector remains unchanged." << std::endl;
        }
        for (float& element : current_vector) {
            element /= sum;
        }

    }
}

auto end = std::chrono::high_resolution_clock::now();
auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

std::cout << "Total time for " << t_times << " multiplications: " << total_time.count() << " μs" << std::endl;
std::cout << "Average time per multiplication: " << (double)total_time.count() / t_times << " μs" << std::endl;


// No need to check for empty if size check passes and we iterate over indices.

// Print first few elements of the final result
// std::cout << "First 10 elements of multiplication vector:" << std::endl;
// for (int i = 0; i < std::min(10, n); ++i) {
// std::cout << "y[" << i << "] = " << result_vector[i] << std::endl;
// }

if (result_vector.size() != degree.size()) {
    throw std::invalid_argument("Vectors must be of the same size for element-wise division (in-place).");
}

// for (size_t i = 0; i < result_vector.size(); ++i) {
//     if (degree[i] == 0.0f) {
//         // Handle division by zero based on your requirements
//         // Option A: Throw an exception:
//         // throw std::runtime_error("Division by zero encountered at index " + std::to_string(i));
//         // Option B: Assign a default error value:
//         // numerator[i] = 0.0f;
//         // Option C (Default floating-point behavior): Let it become Inf/NaN
//         result_vector[i] /= degree[i];
//         std::cerr << "Warning (in-place): Division by zero at index " << i << ". Result is " << result_vector[i] << std::endl;
//     } else {
//         result_vector[i] /= degree[i];
//     }
// }

// putting sink as 0
// result_vector[sink_index] = 0;
return result_vector;


// float eta_sink_neighbours = 0.0;
// for(int i =  row_ptrG[sink_index]; i<row_ptrG[sink_index+1]; i++){
//     std::cout << col_idxG[i] <<std::endl;
//     eta_sink_neighbours += result_vector[col_idxG[i]];
// }
// float scale_factor = sink_value/(degree[sink_index] - eta_sink_neighbours) ;
// std::cout<<eta_sink_neighbours<<"<-esn  scalefactor->"<<scale_factor<<std::endl;
// for (size_t i = 0; i < result_vector.size(); ++i) {
//     result_vector[i] *= scale_factor;
// }


// // Print first few elements of the final result
// std::cout << "First 10 elements of final result vector:" << std::endl;
// for (int i = 0; i < std::min(10, n); ++i) {
// std::cout << "y[" << i << "] = " << result_vector[i] << std::endl;
// }
// std::cout << "------------------------------------------------------------------" << std::endl;
}
