#include <iostream>
#include <vector>
#include <numeric>
#include <cmath> // For std::sqrt, std::fabsf
#include <cuda_runtime.h>
#include <cusparse.h>

// Macro for CUDA error checking
#define CHECK_CUDA(func) \
    do { \
        cudaError_t status = (func); \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Macro for cuSPARSE error checking
#define CHECK_CUSPARSE(func) \
    do { \
        cusparseStatus_t status = (func); \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Custom CUDA kernel for Sparse Matrix-Vector Multiplication (CSR format)
// y = A * x
__global__ void spmv_csr_kernel(int num_rows, const int* row_ptr, const int* col_idx,
                                 const float* values, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0f;
        int start_idx = row_ptr[row];
        int end_idx = row_ptr[row + 1];

        for (int i = start_idx; i < end_idx; ++i) {
            sum += values[i] * x[col_idx[i]];
        }
        y[row] = sum;
    }
}

// SparseMatrix class definition
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

        // Ensure y has the correct size
        if (y.size() != n) {
            y.resize(n);
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
        if (nnz == 0) {
            // If no non-zero elements, y should be all zeros.
            y.assign(n, 0.0f); // Ensure y is sized and filled with zeros
            return;
        }

        // Ensure y has the correct size before copying results back
        if (y.size() != n) {
            y.resize(n);
        }

        // Copy input vector to device
        CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        // Defensive check: Ensure device pointers are valid before kernel launch
        if (!d_row_ptr || !d_col_idx || !d_values || !d_x || !d_y) {
            std::cerr << "ERROR: One or more device pointers are null in SparseMatrix::multiply_custom before kernel launch!" << std::endl;
            exit(EXIT_FAILURE); // Terminate if critical pointers are invalid
        }

        // Launch custom kernel
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        spmv_csr_kernel<<<grid_size, block_size>>>(n, d_row_ptr, d_col_idx, d_values, d_x, d_y);
        CHECK_CUDA(cudaGetLastError()); // Check for errors from kernel launch
        CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to complete

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void printInfo() {
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "Non-zero elements: " << nnz << std::endl;
        std::cout << "Sparsity: " << (1.0 - (double)nnz / (n * n)) * 100 << "%" << std::endl;
    }

    // Print matrix (CSR components)
    void printMatrixCSR() {
        // Copy data back to host for printing
        CHECK_CUDA(cudaMemcpy(h_row_ptr.data(), d_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(h_col_idx.data(), d_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(h_values.data(), d_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

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

// Function to perform repeated matrix-vector multiplication
std::vector<float> repeatedMatrixVectorMultiplication(SparseMatrix& matrix,
                                                      std::vector<float>& initial_vector,
                                                      std::vector<float>& degree, // Not used in this implementation based on the provided context
                                                      std::vector<int>& row_ptrG, // Not used directly, matrix object holds CSR
                                                      std::vector<int>& col_idxG, // Not used directly, matrix object holds CSR
                                                      int sink_index, int sink_value,
                                                      int t_times, bool use_cusparse) {
    int n = matrix.getSize();
    std::vector<float> current_vector = initial_vector;
    std::vector<float> result_vector(n);

    for (int t = 0; t < t_times; ++t) {
        if (use_cusparse) {
            matrix.multiply_cusparse(current_vector, result_vector);
        } else {
            matrix.multiply_custom(current_vector, result_vector);
        }

        // Apply sink logic
        if (sink_index >= 0 && sink_index < n) {
            result_vector[sink_index] += sink_value;
        }

        // Normalize the vector (if needed, based on typical graph algorithms like PageRank)
        // float sum = std::accumulate(result_vector.begin(), result_vector.end(), 0.0f);
        // if (sum != 0.0f) {
        //     for (int i = 0; i < n; ++i) {
        //         result_vector[i] /= sum;
        //     }
        // }

        current_vector = result_vector; // Update current_vector for the next iteration
    }

    return current_vector;
}

// New function to calculate the Laplacian error
float calculateLaplacianError(int alpha_val, int n, const std::vector<float>& sol,
                              const std::vector<int>& row_ptr, const std::vector<int>& col_idx,
                              const std::vector<float>& values, const std::vector<float>& b,
                              bool use_cusparse) {
    // Input validation (added for robustness)
    if (n <= 0) {
        std::cerr << "Error: Matrix size n must be positive." << std::endl;
        return -1.0f; // Indicate error
    }
    if (sol.size() != n || b.size() != n) {
        std::cerr << "Error: sol and b vectors must have size n." << std::endl;
        return -1.0f;
    }
    if (row_ptr.size() != n + 1) {
        std::cerr << "Error: row_ptr must have size n + 1." << std::endl;
        return -1.0f;
    }
    if (col_idx.size() != values.size()) {
        std::cerr << "Error: col_idx and values must have the same size." << std::endl;
        return -1.0f;
    }
    // If values is empty, it means an empty adjacency matrix. Degree will be 0 for all nodes.
    // The Laplacian will be a zero matrix. This is a valid case but might lead to
    // denominator being zero if b is also a zero vector.

    // 1. Construct SparseMatrix object for the adjacency matrix A
    SparseMatrix matrix_A(n, row_ptr, col_idx, values);

    // 2. Calculate degree vector (diagonal elements of D)
    std::vector<float> degree(n);
    for (int i = 0; i < n; ++i) {
        degree[i] = static_cast<float>(row_ptr[i+1] - row_ptr[i]);
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

    // 10. Return the ratio, handling division by zero
    if (denominator_norm == 0.0f) {
        std::cerr << "Warning: Denominator is zero in Laplacian error calculation. Returning 0.0." << std::endl;
        return 0.0f;
    }

    return numerator_norm / denominator_norm;
}

// New CUDA kernel for calculating x_i = sum(|sol_i - sol_j|)
__global__ void calculate_abs_diff_sum_kernel(int n, const int* row_ptr,
                                              const int* col_idx, const float* sol,
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
std::vector<float> calculateAbsoluteDifferenceSum(int n, const std::vector<float>& sol,
                                                  const std::vector<int>& row_ptr,
                                                  const std::vector<int>& col_idx) {
    // Input validation
    if (n <= 0) {
        std::cerr << "Error: Matrix size n must be positive." << std::endl;
        return {}; // Return empty vector
    }
    if (sol.size() != n) {
        std::cerr << "Error: sol vector must have size n." << std::endl;
        return {};
    }
    if (row_ptr.size() != n + 1) {
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


int main() {
    // Example Usage for repeatedMatrixVectorMultiplication
    int n = 5; // Matrix size 5x5

    // Example Sparse Adjacency Matrix (CSR format)
    // A = [[1, 0, 2, 0, 0],
    //      [0, 3, 0, 4, 0],
    //      [5, 0, 6, 0, 7],
    //      [0, 0, 0, 8, 0],
    //      [0, 9, 0, 0, 10]]
    std::vector<int> row_ptr = {0, 2, 4, 7, 8, 10};
    std::vector<int> col_idx = {0, 2, 1, 3, 0, 2, 4, 3, 1, 4};
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    // Initial vector for repeated multiplication
    std::vector<float> initial_vector = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Dummy degree vector (not used in this specific implementation of repeatedMatrixVectorMultiplication)
    std::vector<float> degree_dummy(n, 0.0f);

    int sink_index = 2; // Example sink node index
    int sink_value = 100; // Value to add to the sink node
    int t_times = 3;    // Number of repetitions

    // Create SparseMatrix object
    SparseMatrix matrix(n, row_ptr, col_idx, values);
    matrix.printInfo();
    matrix.printMatrixCSR();

    std::cout << "\n--- Running repeatedMatrixVectorMultiplication with cuSPARSE ---" << std::endl;
    std::vector<float> result_cusparse_repeated = repeatedMatrixVectorMultiplication(
        matrix, initial_vector, degree_dummy, row_ptr, col_idx, sink_index, sink_value, t_times, true);

    std::cout << "Result after " << t_times << " repetitions (cuSPARSE): [";
    for (size_t i = 0; i < result_cusparse_repeated.size(); ++i) {
        std::cout << result_cusparse_repeated[i] << (i == result_cusparse_repeated.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    std::cout << "\n--- Running repeatedMatrixVectorMultiplication with Custom CUDA Kernel ---" << std::endl;
    std::vector<float> result_custom_repeated = repeatedMatrixVectorMultiplication(
        matrix, initial_vector, degree_dummy, row_ptr, col_idx, sink_index, sink_value, t_times, false);

    std::cout << "Result after " << t_times << " repetitions (Custom Kernel): [";
    for (size_t i = 0; i < result_custom_repeated.size(); ++i) {
        std::cout << result_custom_repeated[i] << (i == result_custom_repeated.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    // Example Usage for calculateLaplacianError
    std::cout << "\n--- Running calculateLaplacianError ---" << std::endl;
    std::vector<float> sol_vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b_vector = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    int alpha_value = 2;

    float error_cusparse = calculateLaplacianError(alpha_value, n, sol_vector, row_ptr, col_idx, values, b_vector, true);
    std::cout << "Laplacian Error (cuSPARSE): " << error_cusparse << std::endl;

    float error_custom = calculateLaplacianError(alpha_value, n, sol_vector, row_ptr, col_idx, values, b_vector, false);
    std::cout << "Laplacian Error (Custom Kernel): " << error_custom << std::endl;

    // Example Usage for calculateAbsoluteDifferenceSum
    std::cout << "\n--- Running calculateAbsoluteDifferenceSum ---" << std::endl;
    std::vector<float> abs_diff_sum_result = calculateAbsoluteDifferenceSum(n, sol_vector, row_ptr, col_idx);
    std::cout << "Absolute Difference Sum (x_i): [";
    for (size_t i = 0; i < abs_diff_sum_result.size(); ++i) {
        std::cout << abs_diff_sum_result[i] << (i == abs_diff_sum_result.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    return 0;
}

