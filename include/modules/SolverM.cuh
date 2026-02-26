// include/modules/module1.cuh
#ifndef MODULE1_CUH
#define MODULE1_CUH
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
#include "../../include/modules/helper.cuh" 
#include <cuda_runtime.h>

// Custom CUDA kernel for SpMV (alternative implementation)
__global__ void spmv_csr_kernel(int n, const int* row_ptr, const int* col_idx, const float* values, const float* x, float* y);

class SparseMatrix {
    private:
        int n;  // Matrix size (n x n)
        int nnz;  // Number of non-zero elements
        
        // Host data (initially for input/copying)
        std::vector<int> h_row_ptr;
        std::vector<int> h_col_idx;
        std::vector<float> h_values;
        
        // Device data
        int* d_row_ptr;
        int* d_col_idx;
        float* d_values;
        float* d_x; // Device input vector for SpMV
        float* d_y; // Device output vector for SpMV
        
        // cuSPARSE handles
        cusparseHandle_t cusparse_handle;
        cusparseSpMatDescr_t mat_desc;
        cusparseDnVecDescr_t vec_x, vec_y;
        
        bool cusparse_initialized;
    
    public:
        // Constructor now takes CSR components directly
        SparseMatrix(int size, const std::vector<int>& row_ptr, 
                     const std::vector<int>& col_idx, const std::vector<float>& values) 
            : n(size), h_row_ptr(row_ptr), h_col_idx(col_idx), h_values(values), 
              cusparse_initialized(false) {
            
            nnz = h_values.size(); // Number of non-zero elements
            
            // Allocate device memory
            CHECK_CUDA(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
            
            // Copy data to device
            CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
            
            // Initialize cuSPARSE
            CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
            
            // Create cuSPARSE matrix descriptor
            CHECK_CUSPARSE(cusparseCreateCsr(&mat_desc, n, n, nnz,
                                            d_row_ptr, d_col_idx, d_values,
                                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
            
            // Create vector descriptors
            CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x, n, d_x, CUDA_R_32F));
            CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y, n, d_y, CUDA_R_32F));
            
            cusparse_initialized = true;
        }
        
        ~SparseMatrix() {
            if (d_row_ptr) cudaFree(d_row_ptr);
            if (d_col_idx) cudaFree(d_col_idx);
            if (d_values) cudaFree(d_values);
            if (d_x) cudaFree(d_x);
            if (d_y) cudaFree(d_y);
            
            if (cusparse_initialized) {
                cusparseDestroySpMat(mat_desc);
                cusparseDestroyDnVec(vec_x);
                cusparseDestroyDnVec(vec_y);
            }
            cusparseDestroy(cusparse_handle);
        }
        
        // Get matrix size
        int getSize() const { return n; }
    
        // Perform SpMV using cuSPARSE (y = A * x)
        void multiply_cusparse(const std::vector<float>& x, std::vector<float>& y) {
            if (!cusparse_initialized) {
                std::cerr << "Matrix not initialized!" << std::endl;
                return;
            }
            
            // Copy input vector to device
            CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            
            // Perform SpMV: y = A * x
            float alpha = 1.0f, beta = 0.0f;
            size_t buffer_size = 0;
            
            // Query buffer size
            CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, mat_desc, vec_x, &beta, vec_y,
                                                  CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
            
            // Allocate buffer
            void* buffer = nullptr;
            if (buffer_size > 0) {
                CHECK_CUDA(cudaMalloc(&buffer, buffer_size));
            }
            
            // Perform the multiplication
            CHECK_CUSPARSE(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, mat_desc, vec_x, &beta, vec_y,
                                       CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
            
            // Copy result back to host
            CHECK_CUDA(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
            
            if (buffer) cudaFree(buffer);
        }
        
        // Perform SpMV using custom CUDA kernel (y = A * x)
        void multiply_custom(const std::vector<float>& x, std::vector<float>& y) {
            if (nnz == 0) return;
            
            // Copy input vector to device
            CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            
            // Launch custom kernel
            int block_size = 256;
            int grid_size = (n + block_size - 1) / block_size;
            
            spmv_csr_kernel<<<grid_size, block_size>>>(n, d_row_ptr, d_col_idx, d_values, d_x, d_y);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            
            // Copy result back to host
            CHECK_CUDA(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
        }
    
        // Scalar-Matrix Multiplication: A = t * A (element-wise on non-zeros)
        // void scalarMultiplyMatrix(float t) {
        //     int block_size = 256;
        //     int grid_size = (nnz + block_size - 1) / block_size;
        //     scalar_multiply_kernel<<<grid_size, block_size>>>(nnz, t, d_values);
        //     CHECK_CUDA(cudaGetLastError());
        //     CHECK_CUDA(cudaDeviceSynchronize());
    
        //     // Update the cuSPARSE matrix descriptor to reflect the new values
        //     // Destroy the old descriptor first to avoid memory leaks
        //     cusparseDestroySpMat(mat_desc); 
        //     CHECK_CUSPARSE(cusparseCreateCsr(&mat_desc, n, n, nnz,
        //                                     d_row_ptr, d_col_idx, d_values,
        //                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        //                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        // }
        
        void printInfo() {
            std::cout << "Matrix size: " << n << "x" << n << std::endl;
            std::cout << "Non-zero elements: " << nnz << std::endl;
            std::cout << "Sparsity: " << (1.0 - (double)nnz / (n * n)) * 100 << "%" << std::endl;
        }
    
        // Print matrix (CSR components)
        void printMatrixCSR() {
            // Copy data back to host for printing
            CHECK_CUDA(cudaMemcpy(h_row_ptr.data(), d_row_ptr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_col_idx.data(), d_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_values.data(), d_values, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    
            std::cout << "\n--- Matrix in CSR Format ---" << std::endl;
            std::cout << "row_ptr: [";
            for (size_t i = 0; i < h_row_ptr.size(); ++i) {
                std::cout << h_row_ptr[i] << (i == h_row_ptr.size() - 1 ? "" : ", ");
            }
            std::cout << "]" << std::endl;
    
            std::cout << "col_idx: [";
            for (size_t i = 0; i < h_col_idx.size(); ++i) {
                std::cout << h_col_idx[i] << (i == h_col_idx.size() - 1 ? "" : ", ");
            }
            std::cout << "]" << std::endl;
    
            std::cout << "values:  [";
            for (size_t i = 0; i < h_values.size(); ++i) {
                std::cout << h_values[i] << (i == h_values.size() - 1 ? "" : ", ");
            }
            std::cout << "]" << std::endl;
            std::cout << "----------------------------" << std::endl;
        }
    };
    




// Repeated multiplication of result vector t times (y = A * (A * (... (A * x)...)))
std::vector<float> repeatedMatrixVectorMultiplication(SparseMatrix& matrix, 
    std::vector<float>& initial_vector, 
    std::vector<float>& degree, 
    int t_times, bool use_cusparse); 



#endif // MODULE1_CUH