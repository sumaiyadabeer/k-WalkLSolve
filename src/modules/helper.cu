// src/modules/module2.cu
#include "../../include/modules/helper.cuh"
#include "../../include/config.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <chrono>
#include <numeric>   // For std::iota
#include <fstream>   // For file operations
#include <sstream>   // For string stream parsing
#include <string>    // For string manipulation


// Function to read CSR matrix data from a file
bool readPMatrixFromCSRFile(const std::string& filename, int& n, 
    std::vector<int>& row_ptr, 
    std::vector<int>& col_idx, 
    std::vector<float>& values,
    std::vector<float>& start_dis,
    int& sink_index,
    float& sink_value) 
{std::ifstream infile(filename);
if (!infile.is_open()) {
std::cerr << "Error: Could not open file " << filename << std::endl;
return false;
}

std::string line;
int nnz;

// Read n and nnz
if (std::getline(infile, line)) {
std::stringstream ss(line);
if (!(ss >> n >> nnz)) {
std::cerr << "Error: Invalid format for n and nnz in file " << filename << std::endl;
return false;
}
} else {
std::cerr << "Error: Empty file or cannot read n and nnz from " << filename << std::endl;
return false;
}

// Read row_ptr
if (std::getline(infile, line)) {
std::stringstream ss(line);
row_ptr.resize(n + 1);
for (int i = 0; i <= n; ++i) {
if (!(ss >> row_ptr[i])) {
std::cerr << "Error: Invalid format for row_ptr in file " << filename << std::endl;
return false;
}
}
} else {
std::cerr << "Error: Cannot read row_ptr from " << filename << std::endl;
return false;
}

// Read col_idx
if (std::getline(infile, line)) {
std::stringstream ss(line);
col_idx.resize(nnz);
for (int i = 0; i < nnz; ++i) {
if (!(ss >> col_idx[i])) {
std::cerr << "Error: Invalid format for col_idx in file " << filename << std::endl;
return false;
}
}
} else {
std::cerr << "Error: Cannot read col_idx from " << filename << std::endl;
return false;
}

// Read values
if (std::getline(infile, line)) {
std::stringstream ss(line);
values.resize(nnz);
for (int i = 0; i < nnz; ++i) {
if (!(ss >> values[i])) {
std::cerr << "Error: Invalid format for values in file " << filename << std::endl;
return false;
}
}
} else {
std::cerr << "Error: Cannot read values from " << filename << std::endl;
return false;
}

// Read init_distribution_from degree
if (std::getline(infile, line)) {
std::stringstream ss(line);
start_dis.resize(n);
for (int i = 0; i < n; ++i) {
if (!(ss >> start_dis[i])) {
std::cerr << "Error: Invalid format for start_dis in file " << filename << std::endl;
return false;
}
}
} else {
std::cerr << "Error: Cannot read  start_dis from " << filename << std::endl;
return false;
}

// Read sink_index and sink_value from degree
if (std::getline(infile, line)) {
    std::stringstream ss(line);
    // start_dis.resize(n);
    if (!(ss >> sink_index)) {
        std::cerr << "Error: Invalid format for sink_index and sink_value in file " << filename << std::endl;
        return false;
    }
    if (!(ss >> sink_value)) {
        std::cerr << "Error: Invalid format for sink_index and sink_value in file " << filename << std::endl;
        return false;
    }

    } else {
    std::cerr << "Error: Cannot read  start_dis from " << filename << std::endl;
    return false;
    }
    

infile.close();
std::cout << "Successfully read matrix from " << filename << std::endl;
return true;
}


// Function to read CSR matrix of Graph data from a file
bool readGMatrixFromCSRFile(const std::string& filename, int& n, 
    std::vector<int>& row_ptr, 
    std::vector<int>& col_idx, 
    std::vector<float>& values,
    std::vector<float>& b) 
{std::ifstream infile(filename);
if (!infile.is_open()) {
std::cerr << "Error: Could not open file " << filename << std::endl;
return false;
}

std::string line;
int nnz;

// Read n and nnz
if (std::getline(infile, line)) {
std::stringstream ss(line);
if (!(ss >> n >> nnz)) {
std::cerr << "Error: Invalid format for n and nnz in file " << filename << std::endl;
return false;
}
} else {
std::cerr << "Error: Empty file or cannot read n and nnz from " << filename << std::endl;
return false;
}

// Read row_ptr
if (std::getline(infile, line)) {
std::stringstream ss(line);
row_ptr.resize(n + 1);
for (int i = 0; i <= n; ++i) {
if (!(ss >> row_ptr[i])) {
std::cerr << "Error: Invalid format for row_ptr in file " << filename << std::endl;
return false;
}
}
} else {
std::cerr << "Error: Cannot read row_ptr from " << filename << std::endl;
return false;
}

// Read col_idx
if (std::getline(infile, line)) {
std::stringstream ss(line);
col_idx.resize(nnz);
for (int i = 0; i < nnz; ++i) {
if (!(ss >> col_idx[i])) {
std::cerr << "Error: Invalid format for col_idx in file " << filename << std::endl;
return false;
}
}
} else {
std::cerr << "Error: Cannot read col_idx from " << filename << std::endl;
return false;
}

// Read values
if (std::getline(infile, line)) {
std::stringstream ss(line);
values.resize(nnz);
for (int i = 0; i < nnz; ++i) {
if (!(ss >> values[i])) {
std::cerr << "Error: Invalid format for values in file " << filename << std::endl;
return false;
}
}
} else {
std::cerr << "Error: Cannot read values from " << filename << std::endl;
return false;
}

// Read init_distribution_from degree
if (std::getline(infile, line)) {
std::stringstream ss(line);
b.resize(n);
for (int i = 0; i < n; ++i) {
if (!(ss >> b[i])) {
std::cerr << "Error: Invalid format for start_dis in file " << filename << std::endl;
return false;
}
}
} else {
std::cerr << "Error: Cannot read  start_dis from " << filename << std::endl;
return false;
}


infile.close();
std::cout << "Successfully read matrix from " << filename << std::endl;
return true;
}
// New CUDA kernel for calculating x_i = sum(|sol_i - sol_j|)
__global__ void calculate_abs_diff_sum_kernel(int n,  int* row_ptr,
    int* col_idx,  float* sol,
   float* x_out) {
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < n) {
float sum_abs_diff = 0.0f;
int start_idx = row_ptr[i];
int end_idx = row_ptr[i + 1];

for (int k = start_idx; k < end_idx; ++k) {
int j = col_idx[k]; // Neighbor index
sum_abs_diff += fabsf(sol[i] - sol[j]);
}
x_out[i] = sum_abs_diff;
}
}

// Host-side wrapper function to calculate x_i = sum(|sol_i - sol_j|)
std::vector<float> calculateAbsoluteDifferenceSum(int n, std::vector<float>& sol,
        std::vector<int>& row_ptr,
        std::vector<int>& col_idx) {
// Input validation
if (n <= 0) {
std::cerr << "Error: Matrix size n must be positive." << std::endl;
return {}; // Return empty vector
}
if ((int)sol.size() != n) {
std::cerr << "Error: sol vector must have size n." << std::endl;
return {};
}
if ((int)row_ptr.size() != n + 1) {
std::cerr << "Error: row_ptr must have size n + 1." << std::endl;
return {};
}
// A more robust check for col_idx size would involve iterating through row_ptr
// to determine the total number of non-zeros (nnz) and comparing with col_idx.size().
// For simplicity, we'll assume consistency if row_ptr is valid.

float* d_sol;
int* d_row_ptr;
int* d_col_idx;
float* d_x_out;

// Allocate device memory
CHECK_CUDA(cudaMalloc(&d_sol, n * sizeof(float)));
CHECK_CUDA(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
CHECK_CUDA(cudaMalloc(&d_col_idx, col_idx.size() * sizeof(int))); // Use actual nnz for col_idx
CHECK_CUDA(cudaMalloc(&d_x_out, n * sizeof(float)));

// Copy data to device
CHECK_CUDA(cudaMemcpy(d_sol, sol.data(), n * sizeof(float), cudaMemcpyHostToDevice));
CHECK_CUDA(cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
CHECK_CUDA(cudaMemcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));

// Launch kernel
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;
calculate_abs_diff_sum_kernel<<<grid_size, block_size>>>(n, d_row_ptr, d_col_idx, d_sol, d_x_out);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());

// Copy result back to host
std::vector<float> x_result(n);
CHECK_CUDA(cudaMemcpy(x_result.data(), d_x_out, n * sizeof(float), cudaMemcpyDeviceToHost));

// Free device memory
CHECK_CUDA(cudaFree(d_sol));
CHECK_CUDA(cudaFree(d_row_ptr));
CHECK_CUDA(cudaFree(d_col_idx));
CHECK_CUDA(cudaFree(d_x_out));

return x_result;
}


// New function to calculate the Laplacian error
float calculateLaplacianError(int sink_index, int n, std::vector<float>& sol,
   const std::vector<int>& row_ptr, const std::vector<int>& col_idx,
   const std::vector<float>& values, const std::vector<float>& b,
   bool use_cusparse) {
   // Input validation (added for robustness)
   if (n <= 0) {
   std::cerr << "Error: Matrix size n must be positive." << std::endl;
   return -1.0f; // Indicate error
   }
   if ((int) sol.size() != n || (int) b.size() != n) {
   std::cerr << "Error: sol and b vectors must have size n." << std::endl;
   return -1.0f;
   }
   if ((int)row_ptr.size() != n + 1) {
   std::cerr << "Error: row_ptr must have size n + 1 " <<row_ptr.size() << " is not equal to " <<n+1<< std::endl;
   return -1.0f;
   }
   if (col_idx.size() != values.size()) {
   std::cerr << "Error: col_idx and values must have the same size." << std::endl;
   return -1.0f;
   }
   // If values is empty, it means an empty adjacency matrix. Degree will be 0 for all nodes.
   // The Laplacian will be a zero matrix. This is a valid case but might lead to
   // denominator being zero if b is also a zero vector.

   // for (int i = 0; i < std::min(10, n); ++i) {
   //     // std::cout << "b[" << i << "] = " << b[i] << std::endl;
   // }


   // 1. Construct SparseMatrix object for the adjacency matrix A
   SparseMatrix matrix_A(n, row_ptr, col_idx, values);
   // matrix_A.printMatrixCSR();
   // 2. Calculate degree vector (diagonal elements of D)
   std::vector<float> degree(n);
   for (int i = 0; i < n; ++i) {
   degree[i] = static_cast<float>(row_ptr[i+1] - row_ptr[i]);
//    sol[i] = sol[i]/degree[i]; //converting eta to x
   }
//    convert eta to x 
   sol[sink_index] = 0;
    for (size_t i = 0; i < sol.size(); ++i) {
        if (degree[i] == 0.0f) {
            // Handle division by zero based on your requirements
            // Option A: Throw an exception:
            // throw std::runtime_error("Division by zero encountered at index " + std::to_string(i));
            // Option B: Assign a default error value:
            // numerator[i] = 0.0f;
            // Option C (Default floating-point behavior): Let it become Inf/NaN
            sol[i] /= degree[i];
            std::cerr << "Warning (in-place): Division by zero at index " << i << ". Result is " << sol[i] << std::endl;
        } else {
            sol[i] /= degree[i];
        }
    }
   // 3. Calculate D * sol (element-wise multiplication)
   std::vector<float> D_sol(n);
   for (int i = 0; i < n; ++i) {
   D_sol[i] = degree[i] * sol[i];
   }

   // 4. Calculate A * sol using the SparseMatrix's multiplication method
   std::vector<float> A_sol(n);
   if (use_cusparse) {
   matrix_A.multiply_cusparse(sol, A_sol);
   } else {
   matrix_A.multiply_custom(sol, A_sol);
   }

   // 5. Calculate L * sol = (D * sol) - (A * sol)
   std::vector<float> L_sol(n);
   for (int i = 0; i < n; ++i) {
   L_sol[i] = D_sol[i] - A_sol[i];
   }

   // 6. calculate alpha
   float alpha_val =  L_sol[sink_index]/b[sink_index];
   printf("alpha value %f\n", alpha_val);
   // 6. Calculate alpha * b
   std::vector<float> alpha_b(n);
   for (int i = 0; i < n; ++i) {
   alpha_b[i] = static_cast<float>(alpha_val) * b[i];
   }

   // 7. Calculate the difference vector: (L * sol) - (alpha * b)
   std::vector<float> diff_vec(n);
   for (int i = 0; i < n; ++i) {
   diff_vec[i] = L_sol[i] - alpha_b[i];
   }

   // for (int i = 0; i < n; ++i) {
   //     std::cout << "sol[" << i << "] = " << sol[i] << "	L_sol[" << i << "] = " << L_sol[i] << "	alpha_b[" << i << "] = " << alpha_b[i] << "	diff_vec[" << i << "] = " << diff_vec[i] << std::endl;
   // }
   // 8. Calculate the L2 norm of the numerator: |(L * sol) - (alpha * b)|
   float numerator_norm_sq = 0.0f;
   for (int i = 0; i < n; ++i) {
   numerator_norm_sq += diff_vec[i] * diff_vec[i];
   }
   float numerator_norm = std::sqrt(numerator_norm_sq);

   // 9. Calculate the L2 norm of the denominator: |alpha * b|
   float denominator_norm_sq = 0.0f;
   for (int i = 0; i < n; ++i) {
   denominator_norm_sq += alpha_b[i] * alpha_b[i];
   }
   float denominator_norm = std::sqrt(denominator_norm_sq);

   printf("Numerator_norm: %f\t Denominator_norm: %f \t Realtive_residual: %f\n",numerator_norm, denominator_norm, numerator_norm/denominator_norm);
   // 10. Return the ratio, handling division by zero
   if (denominator_norm == 0.0f) {
   std::cerr << "Warning: Denominator is zero in Laplacian error calculation. Returning 0.0." << std::endl;
   return 0.0f;
   }

   return numerator_norm / denominator_norm;
}

float Wrapper_calculateLaplacianError(int sink_index, int n, int e, float* sol,
   int* row_ptr, int* col_idx, float* b,
   bool use_cusparse){
       // convert to vectors int sink_index, int n, float* sol, int* row_ptr, int* col_idx, float* values, float* b,
       std::vector<float> sol_vec;
       std::vector<int> row_ptr_vec;
       std::vector<int> col_idx_vec;
       // std::vector<float> values_vec;

       std::vector<float> values_vec(e, 1);
       std::vector<float> b_vec;
       sol_vec.assign(sol, sol+n); // n is size of array.
       for (float element : sol_vec) {
        element /= NO_OF_MSJ;
        }
       row_ptr_vec.assign(row_ptr, row_ptr+n+1);
       col_idx_vec.assign(col_idx, col_idx+e);
       // values_vec.assign(values, values+e);
       b_vec.assign(b, b+n);

      
       return calculateLaplacianError(sink_index, n, 
           sol_vec,
           row_ptr_vec,  col_idx_vec, values_vec, 
           b_vec,
           use_cusparse);
   }
