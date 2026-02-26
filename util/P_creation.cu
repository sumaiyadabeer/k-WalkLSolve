#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <chrono>
#include <numeric>   // For std::iota, std::accumulate
#include <fstream>   // For file operations
#include <sstream>   // For string stream parsing
#include <string>    // For string manipulation
#include <iomanip>   // For std::fixed and std::setprecision
#include <algorithm> // For std::max, std::sort
#include <map>       // For building the modified row efficiently

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

/**
 * @brief Checks if a sparse matrix in CSR format is symmetric WITHOUT explicit transpose.
 * A matrix A is symmetric if A = A^T. This function compares elements A[i][j] with A[j][i].
 * Assumes 0-based indexing for CSR data.
 *
 * This version specifically handles a non-standard CSR where the last element of h_row_ptr_A
 * (h_row_ptr_A[num_rows_A]) might be (actual_nnz + 1), meaning it points one past the
 * end of the h_col_idx_A and h_values_A arrays, whose actual size is given by nnz_A.
 *
 * @param num_rows_A Number of rows in matrix A.
 * @param num_cols_A Number of columns in matrix A.
 * @param nnz_A The TRUE number of non-zero elements in matrix A (i.e., size of h_col_idx_A and h_values_A).
 * @param h_row_ptr_A Host array for row_ptr of A.
 * @param h_col_idx_A Host array for col_idx of A.
 * @param h_values_A Host array for values of A.
 * @param tolerance Floating-point tolerance for value comparison.
 * @return true if the matrix is symmetric, false otherwise.
 */
 bool isSymmetricCsrWithoutTranspose(int num_rows_A, int num_cols_A, int nnz_A,
    const std::vector<int>& h_row_ptr_A,
    const std::vector<int>& h_col_idx_A,
    const std::vector<float>& h_values_A,
    float tolerance = 1e-6f) {

// 1. A symmetric matrix must be square.
if (num_rows_A != num_cols_A) {
std::cout << "Matrix is not square (" << num_rows_A << "x" << num_cols_A << "). Cannot be symmetric." << std::endl;
return false;
}

// Sanity check: Ensure h_row_ptr_A is correctly sized
if (h_row_ptr_A.size() != num_rows_A + 1) {
std::cerr << "Error: h_row_ptr_A size is incorrect. Expected " << num_rows_A + 1 << ", got " << h_row_ptr_A.size() << std::endl;
return false;
}

// IMPORTANT CHANGE: Trust nnz_A parameter for array bounds, not h_row_ptr_A[num_rows_A]
// The previous check would return an error if h_row_ptr_A[num_rows_A] was (nnz_A + 1)
// Now, we specifically allow h_row_ptr_A[num_rows_A] to be nnz_A + 1, but cap access at nnz_A.
if (h_col_idx_A.size() < nnz_A || h_values_A.size() < nnz_A) {
std::cerr << "Error: h_col_idx_A or h_values_A size is too small. Expected at least " << nnz_A << ", but got " << h_values_A.size() << std::endl;
return false;
}

// Use a map to store elements from the upper triangular part or track unmatched elements.
std::map<std::pair<int, int>, float> unmatched_elements;

for (int i = 0; i < num_rows_A; ++i) {
// IMPORTANT CHANGE: Adjust the loop upper bound for 'k'
// This ensures 'k' never goes out of bounds of h_col_idx_A and h_values_A,
// even if h_row_ptr_A[i+1] is (nnz_A + 1) for the last row.
int row_elements_end_idx = std::min(h_row_ptr_A[i+1], nnz_A);

for (int k = h_row_ptr_A[i]; k < row_elements_end_idx; ++k) {
int j = h_col_idx_A[k];
float val = h_values_A[k];

// Basic check for valid column index
if (j < 0 || j >= num_cols_A) {
std::cerr << "Error: Invalid column index " << j << " found at row " << i << ", k=" << k << std::endl;
return false;
}

if (i == j) {
// Diagonal element. It must equal itself. No explicit check needed.
continue;
} else if (i < j) {
// Element A[i][j] in the upper triangular part.
// Store its "transpose" counterpart (j,i) with its value.
if (unmatched_elements.count({j, i})) {
std::cerr << "Warning: Duplicate or unexpected entry for (" << j << "," << i << ") encountered for A[" << i << "][" << j << "]." << std::endl;
}
unmatched_elements[{j, i}] = val; // Store A[i][j] at (j,i) in the map
} else { // i > j
// Element A[i][j] in the lower triangular part.
// Look for its transpose counterpart A[j][i] which should have been
// encountered and stored (as (i,j) which maps to (j,i)) in the map.
auto it = unmatched_elements.find({i, j}); // Look for A[j][i] (stored as (i,j) key)

if (it != unmatched_elements.end()) {
// Found the corresponding upper triangular element A[j][i]
// Compare values: A[i][j] vs A[j][i]
if (std::fabs(val - it->second) > tolerance) {
std::cout << "Value mismatch: A[" << i << "][" << j << "] ("
  << std::fixed << std::setprecision(6) << val
  << ") != A[" << j << "][" << i << "] ("
  << std::fixed << std::setprecision(6) << it->second << ")" << std::endl;
return false; // Values are different
}
// Values match, remove it from the map as it's now "matched"
unmatched_elements.erase(it);
} else {
// Element A[i][j] is non-zero, but A[j][i] is implicitly zero (not found in map).
// This means A[i][j] exists but A[j][i] does not have a corresponding non-zero entry.
std::cout << "Structural mismatch: A[" << i << "][" << j << "] ("
<< std::fixed << std::setprecision(6) << val
<< ") exists, but A[" << j << "][" << i << "] is implicitly zero." << std::endl;
return false;
}
}
}
}

// After iterating through all elements, if the map is not empty,
// it means there are elements in the upper triangular part that have
// no corresponding non-zero element in the lower triangular part.
if (!unmatched_elements.empty()) {
std::cout << "Remaining unmatched elements from upper triangle. Matrix is not symmetric." << std::endl;
for (const auto& pair : unmatched_elements) {
std::cout << "Unmatched: A[" << pair.first.second << "][" << pair.first.first << "] (expected A[" << pair.first.first << "][" << pair.first.second << "]) with value " << std::fixed << std::setprecision(6) << pair.second << std::endl;
}
return false;
}

return true; // All checks passed
}



// Custom CUDA kernel for scalar-matrix multiplication (element-wise on non-zeros)
__global__ void scalar_multiply_kernel(int nnz, float scalar, float* values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        values[idx] *= scalar;
    }
}

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
    void scalarMultiplyMatrix(float t) {
        int block_size = 256;
        int grid_size = (nnz + block_size - 1) / block_size;
        scalar_multiply_kernel<<<grid_size, block_size>>>(nnz, t, d_values);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Update the cuSPARSE matrix descriptor to reflect the new values
        // Destroy the old descriptor first to avoid memory leaks
        cusparseDestroySpMat(mat_desc); 
        CHECK_CUSPARSE(cusparseCreateCsr(&mat_desc, n, n, nnz,
                                        d_row_ptr, d_col_idx, d_values,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }
    
    void printInfo() {
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "Non-zero elements: " << nnz << std::endl;
        // Avoid division by zero if n is 0.
        if (n > 0) {
             std::cout << "Sparsity: " << std::fixed << std::setprecision(2) 
                       << (1.0 - (double)nnz / (n * n)) * 100 << "%" << std::endl;
        } else {
             std::cout << "Sparsity: N/A (matrix size 0)" << std::endl;
        }
    }

    // Print matrix (CSR components)
    void printMatrixCSR() {
        // Copy data back to host for printing
        CHECK_CUDA(cudaMemcpy(h_row_ptr.data(), d_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(h_col_idx.data(), d_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(h_values.data(), d_values, nnz * sizeof(float), cudaMemcpyHostToHost)); // Note: HostToHost if already on host
        
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
            std::cout << std::fixed << std::setprecision(6) << h_values[i] << (i == h_values.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
        std::cout << "----------------------------" << std::endl;
    }
};

// Function to read CSR matrix data and an input vector from a file
bool readMatrixAndVectorFromCSRFile(const std::string& filename, int& n, int& num_edges,
                                    std::vector<int>& row_ptr, 
                                    std::vector<int>& col_idx, 
                                    std::vector<float>& values,
                                    std::vector<float>& input_vector) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    int nnz; // This will be the same as num_edges

    // Line 1: num_of_node num_of_edges
    if (std::getline(infile, line)) {
        std::stringstream ss(line);
        if (!(ss >> n >> num_edges)) {
            std::cerr << "Error: Invalid format for num_of_node and num_of_edges in file " << filename << std::endl;
            return false;
        }
        nnz = num_edges; // In CSR, nnz is the number of edges
    } else {
        std::cerr << "Error: Empty file or cannot read num_of_node and num_of_edges from " << filename << std::endl;
        return false;
    }

    // Line 2: row_ptr
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

    // Line 3: col_offset
    if (std::getline(infile, line)) {
        std::stringstream ss(line);
        col_idx.resize(nnz);
        for (int i = 0; i < nnz; ++i) {
            if (!(ss >> col_idx[i])) {
                std::cerr << "Error: Invalid format for col_offset (col_idx) in file " << filename << std::endl;
                return false;
            }
        }
    } else {
        std::cerr << "Error: Cannot read col_offset (col_idx) from " << filename << std::endl;
        return false;
    }

    // Line 4: values
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

    // Line 5: num_of_node values for the input vector
    if (std::getline(infile, line)) {
        std::stringstream ss(line);
        input_vector.resize(n);
        for (int i = 0; i < n; ++i) {
            if (!(ss >> input_vector[i])) {
                std::cerr << "Error: Invalid format for input vector values in file " << filename << std::endl;
                return false;
            }
        }
    } else {
        std::cerr << "Error: Cannot read input vector from " << filename << std::endl;
        return false;
    }

    infile.close();
    std::cout << "Successfully read graph and vector from " << filename << std::endl;
    return true;
}

// Function to transform A to P = A D^-1 (where D is based on column sums of A)
// and modify column 'negative_idx'
void transformGraphAndModifyColumn(int n, const std::vector<int>& A_row_ptr,
                                   const std::vector<int>& A_col_idx,
                                   const std::vector<float>& A_values,
                                   const std::vector<float>& b, int negative_idx,
                                   std::vector<int>& P_row_ptr,
                                   std::vector<int>& P_col_idx,
                                   std::vector<float>& P_values) {

    const float TOLERANCE = 1e-9f; // Tolerance for floating point comparisons to zero

    // Step 1: Calculate column sums (degrees) for matrix A
    // We need to iterate through A to get column sums since A is in CSR format
    std::vector<float> col_sums(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int k = A_row_ptr[i]; k < A_row_ptr[i+1]; ++k) {
            col_sums[A_col_idx[k]] += A_values[k];
        }
    }

    // Prepare for building P's CSR
    std::vector<std::vector<std::pair<int, float>>> P_rows(n); // To store (col_idx, value) for each row
    int current_nnz = 0;

    float neg_b_idx_val = -b[negative_idx];

    if (std::abs(neg_b_idx_val) < TOLERANCE) {
        std::cerr << "Error: Value in b at index " << negative_idx << " is too close to zero. Cannot divide." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Step 2: Construct P and perform column modification
    for (int r = 0; r < n; ++r) { // Iterate through rows of A
        for (int k = A_row_ptr[r]; k < A_row_ptr[r+1]; ++k) { // Iterate through non-zero entries in current row
            int c = A_col_idx[k];
            float original_value = A_values[k];
            float p_rc;

            if (c == negative_idx) { // This is the column that needs modification
                if (r == negative_idx) {
                    p_rc = 0.0f; // P_ii = 0
                } else {
                    p_rc = std::max(0.0f, b[r] / neg_b_idx_val);
                }
            } else { // Standard P = A D^-1 calculation for other columns
                float divisor = col_sums[c];
                if (std::abs(divisor) < TOLERANCE) {
                    p_rc = 0.0f; // If column sum is zero, corresponding P entry will be zero
                } else {
                    p_rc = original_value / divisor;
                }
            }

            if (std::abs(p_rc) > TOLERANCE) { // Only add non-zero elements
                P_rows[r].push_back({c, p_rc});
                current_nnz++;
            }
        }
    }

    // After processing all original A entries, we need to handle potential new entries
    // in the modified column for rows where A had no entry in that column.
    // This is crucial because modifying a column might introduce non-zero elements
    // in rows that previously had a zero at 'negative_idx' column.
    for (int r = 0; r < n; ++r) {
        if (r != negative_idx) { // Don't re-process the diagonal element for the modified column
            bool found_in_original_A = false;
            for (int k = A_row_ptr[r]; k < A_row_ptr[r+1]; ++k) {
                if (A_col_idx[k] == negative_idx) {
                    found_in_original_A = true;
                    break;
                }
            }

            if (!found_in_original_A) {
                float p_rc = std::max(0.0f, b[r] / neg_b_idx_val);
                if (std::abs(p_rc) > TOLERANCE) {
                    P_rows[r].push_back({negative_idx, p_rc});
                    current_nnz++;
                }
            }
        }
    }

    // Sort entries within each row by column index and populate final P_col_idx and P_values
    P_row_ptr.assign(n + 1, 0);
    P_col_idx.reserve(current_nnz);
    P_values.reserve(current_nnz);

    int nnz_count = 0;
    for (int r = 0; r < n; ++r) {
        P_row_ptr[r] = nnz_count;
        std::sort(P_rows[r].begin(), P_rows[r].end()); // Sort by column index

        for (const auto& entry : P_rows[r]) {
            P_col_idx.push_back(entry.first);
            P_values.push_back(entry.second);
            nnz_count++;
        }
    }
    P_row_ptr[n] = nnz_count;
}
// Usage:: nvcc P_creation.cu -lcublas -lcusparse -o P_creation
int main() {
    int n;          // Number of nodes (matrix size)
    int num_edges;  // Number of edges (number of non-zero elements in A)

    std::cout << "CUDA Graph Matrix Transformation Demo" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Check CUDA device availability
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    
    // 1. Read graph data (CSR format) and initial vector from a file
    std::string filename;
    std::cout << "\nEnter the input file name (e.g., graph_input.txt): ";
    std::cin >> filename;

    std::vector<int> A_row_ptr;
    std::vector<int> A_col_idx;
    std::vector<float> A_values;
    std::vector<float> b_vector; // The input vector from Line 5

    if (!readMatrixAndVectorFromCSRFile(filename, n, num_edges, A_row_ptr, A_col_idx, A_values, b_vector)) {
        return -1; // Exit if file reading fails
    }

    // Find the index 'i' of the negative entry in vector b
    int negative_idx = -1;
    float negative_value = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (b_vector[i] < 0) {
            if (negative_idx != -1) {
                std::cerr << "Error: Input vector 'b' has more than one negative entry. Please ensure only one." << std::endl;
                return -1;
            }
            negative_idx = i;
            negative_value = b_vector[i];
        }
    }

    if (negative_idx == -1) {
        std::cerr << "Error: No negative entry found in the input vector 'b'." << std::endl;
        return -1;
    }


    std::cout << "\nInput 'row ptr' (first 10 elements): Size: " << A_row_ptr.size() << std::endl;
    for (int i = 0; i < std::min(10, n); ++i) {
        std::cout << "row ptr[" << i << "] = " << std::fixed <<  A_row_ptr[i] << std::endl;
    }

    std::cout << "\nInput 'col_off' (first 10 elements): Size: " << A_col_idx.size() << std::endl;
    for (int i = 0; i < std::min(10, n); ++i) {
        std::cout << "col_off[" << i << "] = " << std::fixed << A_col_idx[i] << std::endl;
    }

    std::cout << "\nInput 'values' (first 10 elements): Size: " << A_values.size() << std::endl;
    for (int i = 0; i < std::min(10, n); ++i) {
        std::cout << "values[" << i << "] = " << std::fixed << std::setprecision(6) << A_values[i] << std::endl;
    }
    std::cout << "\nInput vector 'b' (first 10 elements): Size: " << b_vector.size() << std::endl;
    for (int i = 0; i < std::min(10, n); ++i) {
        std::cout << "b[" << i << "] = " << std::fixed << std::setprecision(6) << b_vector[i] << std::endl;
    }
    
    std::cout << "Checking Symmetric Matrix Example:" << std::endl;
    if (isSymmetricCsrWithoutTranspose(n, n, num_edges, A_row_ptr, A_col_idx, A_values)) {
        std::cout << "Matrix is symmetric." << std::endl;
        // 2. Transform A to P = D^-1 A and modify row 'negative_idx'
    std::vector<int> P_row_ptr;
    std::vector<int> P_col_idx;
    std::vector<float> P_values;
    
    std::cout << "Transform start" << std::endl;
    transformGraphAndModifyColumn(n, A_row_ptr, A_col_idx, A_values, b_vector, negative_idx,
                               P_row_ptr, P_col_idx, P_values);
    std::cout << "Transform end and sparse matrix creation start" << std::endl;
    // Create sparse matrix object for the transformed matrix P (for printing info/CSR)
    std::cout << "\nInput 'row ptr' (first 10 elements): Size: " << P_row_ptr.size() << std::endl;
    for (int i = 0; i < std::min(10, n); ++i) {
        std::cout << "row ptr[" << i << "] = " << std::fixed <<  P_row_ptr[i] << std::endl;
    }

    std::cout << "\nInput 'col_off' (first 10 elements): Size: " << P_col_idx.size() << std::endl;
    for (int i = 0; i < std::min(10, n); ++i) {
        std::cout << "col_off[" << i << "] = " << std::fixed << P_col_idx[i] << std::endl;
    }

    std::cout << "\nInput 'values' (first 10 elements): Size: " << P_values.size() << std::endl;
    for (int i = 0; i < std::min(10, n); ++i) {
        std::cout << "values[" << i << "] = " << std::fixed << std::setprecision(6) << P_values[i] << std::endl;
    }
  
    


    std::ofstream fout(filename + "_P");

    // Output: Line 1-4: sparse matrix representation like input of P
    std::cout << "\nOutput Lines 1-4 (Transformed Matrix P in CSR format):" << std::endl;
    // std::cout << n << " " << P_values.size() << std::endl; // num_of_node num_of_edges (nnz for P)
    fout << n << " " << P_values.size() << std::endl; // num_of_node num_of_edges (nnz for P)
    
    for (size_t i = 0; i < P_row_ptr.size(); ++i) {
        // std::cout << P_row_ptr[i] << (i == P_row_ptr.size() - 1 ? "" : " ");
        fout << P_row_ptr[i] << (i == P_row_ptr.size() - 1 ? "" : " ");
    }
    std::cout << std::endl;
    fout << std::endl;
    
    for (size_t i = 0; i < P_col_idx.size(); ++i) {
        // std::cout << P_col_idx[i] << (i == P_col_idx.size() - 1 ? "" : " ");
        fout << P_col_idx[i] << (i == P_col_idx.size() - 1 ? "" : " ");
    }
    std::cout << std::endl;
    fout << std::endl;
    
    for (size_t i = 0; i < P_values.size(); ++i) {
        // std::cout << std::fixed << std::setprecision(6) << P_values[i] << (i == P_values.size() - 1 ? "" : " ");
        fout << std::fixed << std::setprecision(6) << P_values[i] << (i == P_values.size() - 1 ? "" : " ");
    }
    std::cout << std::endl;
    fout << std::endl;

    // ---
    // std::cout << "\n--- Degrees of Nodes in the Original Input Graph ---" << std::endl;
    for (int i = 0; i < n; ++i) {
        int degree = A_row_ptr[i+1] - A_row_ptr[i];
        // std::cout << degree << (i == n - 1 ? "" : " ");
        fout << degree << (i == n - 1 ? "" : " ");
    }
        // Output: Line 5: negative entry of b and negative index
        std::cout << std::endl << negative_idx << std::fixed << std::setprecision(6) << negative_value << " " << std::endl;
        fout << std::endl << negative_idx << std::fixed << std::setprecision(6) << negative_value << " " << std::endl;
    // std::cout << "------------------------------------------------------------------" << std::endl;
    
    std::cout << "Sparse matrix creation start" << std::endl;
    SparseMatrix P_matrix(n, P_row_ptr, P_col_idx, P_values);
    std::cout << "\n--- Transformed Matrix P ((D^-1 A)^T with row modification) ---" << std::endl;
    P_matrix.printInfo();
    } else {
        std::cout << "Matrix is NOT symmetric." << std::endl;
    }
    std::cout << "-------------------------------------------\n" << std::endl;



    return 0;
}  