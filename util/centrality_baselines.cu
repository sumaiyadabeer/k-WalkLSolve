#include <iostream>
#include <vector>
#include <numeric>   // For std::iota
#include <fstream>   // For file operations
#include <sstream>   // For string stream parsing
#include <string>    // For string manipulation
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
// Function to read CSR matrix of Graph data from a file
bool readGMatrixFromCSRFile(const std::string& filename, int& n, 
    std::vector<int>& row_ptr, 
    std::vector<int>& col_idx, 
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
// skipping values
std::getline(infile, line);

// Read b
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

void extract_degrees(const std::vector<int>& row_ptr, std::vector<int>& degrees) {
    degrees.resize(row_ptr.size() - 1);
    for (size_t i = 0; i < degrees.size(); ++i) {
        degrees[i] = row_ptr[i+1] - row_ptr[i];
    }
}

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

void read_vector_from_file(const std::string& filename, std::vector<float>& vec) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    vec.clear(); // Clear the vector in case it's not empty
    std::string line;
    
    while (std::getline(infile, line)) {
        try {
            // Convert each line to a double and add to the vector
            vec.push_back(std::stod(line));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid number format in file: " << line << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Number out of range in file: " << line << std::endl;
        }
    }
    
    infile.close();
}


int main(int argc, char *argv[]){
    std::string Gfilename = argv[1]; 
    std::string Ksol = argv[2];  
    // int centrality_frm_mm_required = std::stoi(argv[3]); //0 if MM centrality is already been calculated
    
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> b;
    int n;



    if (!readGMatrixFromCSRFile(Gfilename, n, row_ptr, col_idx,  b)) {
        return -1; // Exit if file reading fails
    }





    // Kwalk CFB centrality calculation

    std::vector<float> sol_vector;
    std::vector<int> degree;
    extract_degrees(row_ptr,degree);
    int sink_vertex_idx = -1;
    for (size_t i = 0; i < b.size(); ++i) {
        if (b[i] < 0) {
            sink_vertex_idx = i;
            break;
        }
    }
    std::cout<< "sink index is: " <<sink_vertex_idx << std::endl;
    // int sink_index = -neg b; // Example sink node index
    read_vector_from_file(Ksol, sol_vector);

    // Convert eta to x
    // for (size_t i = 0; i < sol_vector.size(); ++i) {
    //     if (degree[i] == 0.0f) {
    //         // numerator[i] = 0.0f; // Option C (Default floating-point behavior): Let it become Inf/NaN
    //         sol_vector[i] /= degree[i];
    //         std::cerr << "Warning (in-place): Division by zero at index " << i << ". sol_vector is " << sol_vector[i] << std::endl;
    //     } else {
    //         sol_vector[i] *= degree[i];
    //     }
    // }
    // sol_vector[sink_vertex_idx] = 0;
    
    // Saving x for centrality
    std::string OpFileKX = Ksol + "_eta";
    std::ofstream optFile(OpFileKX);
    // Check if the file was opened successfully
    if (!optFile.is_open()) {
        std::cerr << "Error: Unable to open the file." << std::endl;
        return 1; // Indicate an error
    }
    std::cout << "Printing to file OpFileKX" << std::endl;
    for (int i = 0; i < sol_vector.size(); ++i) {
        // std::cout<< h_eta[i] <<std::endl;
        optFile <<  sol_vector[i]*degree[i] <<  "\n"; //h_b[i]?1:0 
    }
    std::cout << "Printing to file OpFileKX complete" << std::endl;
    // Close the file
    optFile.close();
    


    // Example Usage for calculateAbsoluteDifferenceSum
    std::cout << "\n--- Running calculateAbsoluteDifferenceSum ---" << std::endl;
    std::vector<float> abs_diff_sum_result = calculateAbsoluteDifferenceSum(n, sol_vector, row_ptr, col_idx);
    std::cout << "Absolute Difference Sum (x_i): [";
    for (size_t i = 0; i < std::min(10, n); ++i) {
        std::cout << abs_diff_sum_result[i] << (i == abs_diff_sum_result.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
    std::string OpFileK = Ksol + "_cfbc";
    std::ofstream outputFile(OpFileK);
    // Check if the file was opened successfully
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open the file." << std::endl;
        return 1; // Indicate an error
    }

    std::cout << "Printing to file OpFileK" << std::endl;
    for (int i = 0; i < abs_diff_sum_result.size(); ++i) {
        // std::cout<< h_eta[i] <<std::endl;
        outputFile <<  abs_diff_sum_result[i] <<  "\n"; //h_b[i]?1:0 
    }
    std::cout << "Printing to file OpFileK complete" << std::endl;
    // Close the file
    outputFile.close();


    // if(centrality_frm_mm_required){
    //     // MM centrality code begins from here
    // read_vector_from_file(Msol, sol_vector);


    // // Saving x \t b for centrality
    // std::string OpFileMB = Msol + "_b";
    // std::ofstream optFile3(OpFileMB);
    // // Check if the file was opened successfully
    // if (!optFile3.is_open()) {
    //     std::cerr << "Error: Unable to open the file." << std::endl;
    //     return 1; // Indicate an error
    // }
    // std::cout << "Printing to file OpFileMB" << std::endl;
    // for (int i = 0; i < sol_vector.size(); ++i) {
    //     // std::cout<< h_eta[i] <<std::endl;
    //     optFile3 <<  sol_vector[i] <<"\t" << b[i]<<   "\n"; //h_b[i]?1:0 
    // }
    // std::cout << "Printing to file OpFileMB complete" << std::endl;
    // // Close the file
    // optFile3.close();

    // // // Saving eta \t b for centrality
    // std::string OpFileM_ETA = Msol + "_eta_b";
    // std::ofstream optFile4(OpFileM_ETA);
    // // Check if the file was opened successfully
    // if (!optFile4.is_open()) {
    //     std::cerr << "Error: Unable to open the file." << std::endl;
    //     return 1; // Indicate an error
    // }
    // std::cout << "Printing to file OpFileM_ETA" << std::endl;
    // for (int i = 0; i < sol_vector.size(); ++i) {
    //     optFile4 <<  sol_vector[i]*degree[i] <<"\t" << b[i]<<   "\n"; //h_b[i]?1:0 
    // }
    // std::cout << "Printing to file OpFileM_ETA complete" << std::endl;
    // // Close the file
    // optFile4.close();


    //     // Example Usage for calculateAbsoluteDifferenceSum
    //     std::cout << "\n--- Running calculateAbsoluteDifferenceSum ---" << std::endl;
    //     abs_diff_sum_result = calculateAbsoluteDifferenceSum(n, sol_vector, row_ptr, col_idx);
    //     std::cout << "Absolute Difference Sum (x_i): [";
    //     for (size_t i = 0; i < std::min(10, n); ++i) {
    //         std::cout << abs_diff_sum_result[i] << (i == abs_diff_sum_result.size() - 1 ? "" : ", ");
    //     }
    //     std::cout << "]" << std::endl;

    //     std::ofstream outputFile2(OpFileM);
    //     // Check if the file was opened successfully
    //     if (!outputFile2.is_open()) {
    //         std::cerr << "Error: Unable to open the file." << std::endl;
    //         return 1; // Indicate an error
    //     }
    //     std::cout << "Printing to file" << std::endl;
    //     for (int i = 0; i < abs_diff_sum_result.size(); ++i) {
    //         outputFile2 <<  abs_diff_sum_result[i] <<"\t" << b[i] << std::endl;
    //     }
    //     std::cout << "Printing to file complete" << std::endl;
    //     // Close the file
    //     outputFile2.close();
    // }else{
    //     std::cout<<"No need to calculate centrlity based on MM \n";
    // }

    return 0;
}

