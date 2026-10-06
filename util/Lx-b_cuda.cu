#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

#define SIZE_IN_BYTES(v) (v.size() * sizeof(v[0]))

__global__ void csr_mat_vec_mult(const int V, const int* row_ptr, const int* col_off, const double* values, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        double sum = 0.0;
        int row_start = row_ptr[i];
        int row_end = row_ptr[i+1];
        for (int j = row_start; j < row_end; ++j) {
            sum += values[j] * x[col_off[j]];
        }
        y[i] = sum;
    }
}

__global__ void axpby(const int V, const double alpha, const double* x, const double beta, const double* y, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        out[i] = alpha * x[i] + beta * y[i];
    }
}

__global__ void diagonal_preconditioner(const int V, const double* M_diag, const double* r, double* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        z[i] = r[i] / M_diag[i];
    }
}

__global__ void laplacian_construction_kernel(const int V, const int* row_ptr_A, double* laplacian_diag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        laplacian_diag[i] = row_ptr_A[i+1] - row_ptr_A[i];
    }
}

__global__ void laplacian_mult_kernel(int V, const double* d_laplacian_diag, const double* d_vec, const double* d_vec_mult_adj, double* d_L_mult_vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        d_L_mult_vec[i] = d_laplacian_diag[i] * d_vec[i] - d_vec_mult_adj[i];
    }
}

__global__ void shift_solution_kernel(const int V, const double* x, double* shifted_x, double shift_amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        shifted_x[i] = x[i] - shift_amount;
    }
}

struct square_functor {
    __host__ __device__
    double operator()(double x) const {
        return x * x;
    }
};

double thrust_dot(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b) {
    return thrust::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double thrust_norm(const thrust::device_vector<double>& vec) {
    return std::sqrt(thrust::transform_reduce(
        vec.begin(),
        vec.end(),
        square_functor(),
        0.0,
        thrust::plus<double>()
    ));
}

int main(int argc, char* argv[]) {
    // --- Parse Command Line Arguments ---
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_csr_file> <output_file>" << std::endl;
        return 1;
    }
    std::string input_file_path = argv[1];
    std::string OpFileName = argv[2];

    // --- 1. Read input file into host memory ---
    std::ifstream infile(input_file_path);
    if (!infile) {
        std::cerr << "Error: Could not open input File: " << input_file_path << std::endl;
        return 1;
    }

    std::string line;
    std::getline(infile, line);
    std::stringstream ss(line);
    int V, E;
    ss >> V >> E;

    std::vector<int> row_ptr_A;
    std::vector<int> col_off_A;
    std::vector<double> values_A;
    std::vector<double> b;
    
    std::getline(infile, line);
    std::stringstream ss_row_ptr(line);
    int temp_int;
    while (ss_row_ptr >> temp_int) {
        row_ptr_A.push_back(temp_int);
    }

    std::getline(infile, line);
    std::stringstream ss_col_off(line);
    while (ss_col_off >> temp_int) {
        col_off_A.push_back(temp_int);
    }

    std::getline(infile, line);
    std::stringstream ss_values(line);
    double temp_double;
    while (ss_values >> temp_double) {
        values_A.push_back(temp_double);
    }

    std::getline(infile, line);
    std::stringstream ss_b(line);
    while (ss_b >> temp_double) {
        b.push_back(temp_double);
    }

    if (row_ptr_A.size() != (size_t)V + 1) {
        std::cerr << "Error: Mismatch in row_ptr_A size." << std::endl;
        return 1;
    }
    if (col_off_A.size() != (size_t)E || values_A.size() != (size_t)E) {
        std::cerr << "Error: Mismatch in col_off_A or values_A size." << std::endl;
        return 1;
    }
    if (b.size() != (size_t)V) {
        std::cerr << "Error: Mismatch in b size." << std::endl;
        return 1;
    }

    std::cout << "Successfully read graph with V=" << V << ", E=" << E << std::endl;

    // --- 2. Allocate and copy data to GPU device memory using Thrust ---
    thrust::device_vector<int> d_row_ptr_A = row_ptr_A;
    thrust::device_vector<int> d_col_off_A = col_off_A;
    thrust::device_vector<double> d_values_A = values_A;
    thrust::device_vector<double> d_b = b;
    thrust::device_vector<double> d_x(V, 0.0);
    thrust::device_vector<double> d_r(V);
    thrust::device_vector<double> d_z(V);
    thrust::device_vector<double> d_p(V);
    thrust::device_vector<double> d_q(V);
    thrust::device_vector<double> d_Lx(V);

    // --- 3. Construct the Laplacian Matrix (L=D-A) components on the GPU ---
    thrust::device_vector<double> d_laplacian_diag(V);
    laplacian_construction_kernel<<< (V + 255) / 256, 256 >>>(V, thrust::raw_pointer_cast(d_row_ptr_A.data()), thrust::raw_pointer_cast(d_laplacian_diag.data()));
    
    // --- 4. PCG Algorithm ---
    const int MAX_ITERS = 10000;
    const double TOLERANCE = 1e-7;
    double b_norm = thrust_norm(d_b);

    d_r = d_b;

    diagonal_preconditioner<<< (V + 255) / 256, 256 >>>(V, thrust::raw_pointer_cast(d_laplacian_diag.data()), thrust::raw_pointer_cast(d_r.data()), thrust::raw_pointer_cast(d_z.data()));
    d_p = d_z;

    double r_dot_z_old = thrust_dot(d_r, d_z);

    for (int k = 0; k < MAX_ITERS; ++k) {
        thrust::device_vector<double> d_Ap(V);
        csr_mat_vec_mult<<< (V + 255) / 256, 256 >>>(V, thrust::raw_pointer_cast(d_row_ptr_A.data()), thrust::raw_pointer_cast(d_col_off_A.data()), thrust::raw_pointer_cast(d_values_A.data()), thrust::raw_pointer_cast(d_p.data()), thrust::raw_pointer_cast(d_Ap.data()));
        laplacian_mult_kernel<<< (V + 255) / 256, 256 >>>(V, thrust::raw_pointer_cast(d_laplacian_diag.data()), thrust::raw_pointer_cast(d_p.data()), thrust::raw_pointer_cast(d_Ap.data()), thrust::raw_pointer_cast(d_q.data()));

        double p_dot_q = thrust_dot(d_p, d_q);
        if (p_dot_q == 0) {
            std::cout << "p_dot_q is zero. Stopping." << std::endl;
            break;
        }
        double alpha = r_dot_z_old / p_dot_q;

        axpby<<< (V + 255) / 256, 256 >>>(V, alpha, thrust::raw_pointer_cast(d_p.data()), 1.0, thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_x.data()));
        axpby<<< (V + 255) / 256, 256 >>>(V, -alpha, thrust::raw_pointer_cast(d_q.data()), 1.0, thrust::raw_pointer_cast(d_r.data()), thrust::raw_pointer_cast(d_r.data()));

        double r_norm = thrust_norm(d_r);
        double rel_residual = r_norm / b_norm;
        if (rel_residual < TOLERANCE) {
            std::cout << "Solver converged at iteration " << k << " with relative residual " << rel_residual << std::endl;
            break;
        }

        diagonal_preconditioner<<< (V + 255) / 256, 256 >>>(V, thrust::raw_pointer_cast(d_laplacian_diag.data()), thrust::raw_pointer_cast(d_r.data()), thrust::raw_pointer_cast(d_z.data()));

        double r_dot_z_new = thrust_dot(d_r, d_z);
        double beta = r_dot_z_new / r_dot_z_old;

        axpby<<< (V + 255) / 256, 256 >>>(V, 1.0, thrust::raw_pointer_cast(d_z.data()), beta, thrust::raw_pointer_cast(d_p.data()), thrust::raw_pointer_cast(d_p.data()));
        
        r_dot_z_old = r_dot_z_new;

        if (k % 100 == 0) {
            std::cout << "Iteration " << k << ", Rel. Residual: " << rel_residual << std::endl;
        }
    }

    // --- 5. Final Relative Residual Calculation ---
    thrust::device_vector<double> d_Lx_temp(V);
    csr_mat_vec_mult<<< (V + 255) / 256, 256 >>>(V, thrust::raw_pointer_cast(d_row_ptr_A.data()), thrust::raw_pointer_cast(d_col_off_A.data()), thrust::raw_pointer_cast(d_values_A.data()), thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_Lx_temp.data()));
    laplacian_mult_kernel<<< (V + 255) / 256, 256 >>>(V, thrust::raw_pointer_cast(d_laplacian_diag.data()), thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_Lx_temp.data()), thrust::raw_pointer_cast(d_Lx.data()));

    thrust::device_vector<double> d_diff(V);
    thrust::transform(d_b.begin(), d_b.end(), d_Lx.begin(), d_diff.begin(), thrust::minus<double>());

    double final_residual = thrust_norm(d_diff) / b_norm;
    std::cout << "\nFinal Relative Residual: " << final_residual << std::endl;

    // --- 6. Shift the solution ---
    int sink_vertex_idx = -1;
    double sink_value = 0.0;
    thrust::host_vector<double> h_x = d_x;

    for (size_t i = 0; i < b.size(); ++i) {
        if (b[i] < 0) {
            sink_vertex_idx = i;
            sink_value = h_x[i];
            break;
        }
    }

    thrust::device_vector<double> d_shifted_x(V);
    shift_solution_kernel<<< (V + 255) / 256, 256 >>>(V, thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_shifted_x.data()), sink_value);
    
    thrust::host_vector<double> h_shifted_x = d_shifted_x;

    // --- 7. Apply Degree Multiplication and Write to Output File ---
    std::ofstream outputFile(OpFileName);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open output file: " << OpFileName << std::endl;
        return 1;
    }

    std::cout << "Writing degree-multiplied solution to: " << OpFileName << std::endl;
    for (size_t i = 0; i < h_shifted_x.size(); ++i) {
        int degree = row_ptr_A[i+1] - row_ptr_A[i];
        double degree_mult_val = h_shifted_x[i] * degree;
        outputFile << degree_mult_val << "\n";
    }
    outputFile.close();
    std::cout << "Writing completed successfully." << std::endl;

    return 0;
}