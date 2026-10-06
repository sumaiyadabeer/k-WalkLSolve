#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>

#include "../inc/b_function.h"
#include "../inc/communication.h"
#include "../inc/eta_function.h"
#include "../inc/helper.h"
#include "../inc/calculate_error.h"

#define THREADS_PER_BLOCK 32

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace std::chrono;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("%s\n",cudaGetErrorString(x)); \
    system("pause"); \
    return EXIT_FAILURE;}} while(0)


__global__ void solve(int *row_ptr, float *b_sink, float *eta, float *beta, float kappa, float *x ){
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
	x[index]=(eta[index]/(row_ptr[index+1]-row_ptr[index])); 
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

// --- METRICS EVALUATION FUNCTION (CPU - ONLY NDCG) ---
float evaluateRoundMetrics(int round, int r, 
                          const std::vector<int>& curr_idx, const std::vector<float>& curr_eta,
                          const std::vector<int>& prev_idx, const std::vector<float>& prev_eta) {
    
    // Build lookup map for previous relevance score
    std::map<int, float> prev_relevance_map;
    for (size_t i = 0; i < prev_idx.size(); i++) {
        prev_relevance_map[prev_idx[i]] = prev_eta[i];
    }

    // 1. Graded DCG Calculation
    float dcg = 0.0f;
    for (int i = 0; i < r; i++) {
        int node = curr_idx[i];
        float relevance = prev_relevance_map.count(node) ? prev_relevance_map[node] : 0.0f;
        dcg += relevance / std::log2(static_cast<float>(i + 2));
    }

    // 2. Ideal DCG (IDCG) Calculation
    std::vector<float> ideal_relevance = prev_eta;
    std::sort(ideal_relevance.begin(), ideal_relevance.end(), std::greater<float>());
    
    float idcg = 0.0f;
    for (int i = 0; i < r; i++) {
        idcg += ideal_relevance[i] / std::log2(static_cast<float>(i + 2));
    }

    float ndcg = (idcg > 0.0f) ? (dcg / idcg) : 0.0f;

    // Output formatted NDCG
    std::cout << " -> Round " << round << " | Top-R (" << r << ") Graded NDCG: " 
              << std::fixed << std::setprecision(5) << ndcg << std::endl;

    return ndcg;
}

// 2.5.3. Thrust norms
struct square { __device__ __host__ float operator()(float xi) { return xi*xi; } };
struct absolute { __device__ __host__ float operator()(float xi) { return abs(xi); } };
struct int_absolute { __device__ __host__ int operator()(int xi) { return abs(xi); } };
float two_norm(const thrust::device_ptr<float> &x, int N){
	thrust::device_ptr<float> x_end = x + (N);
	return sqrt(thrust::reduce(
	thrust::make_transform_iterator(x, square()),
	thrust::make_transform_iterator(x_end, square())));
}
float one_norm(const thrust::device_ptr<float> &x, int N){
	thrust::device_ptr<float> x_end = x + (N);
	return thrust::reduce(
	thrust::make_transform_iterator(x, absolute()),
	thrust::make_transform_iterator(x_end, absolute()));
}
int int_one_norm(const thrust::device_ptr<int> &x, int N){
	thrust::device_ptr<int> x_end = x + (N);
	return thrust::reduce(
	thrust::make_transform_iterator(x, int_absolute()),
	thrust::make_transform_iterator(x_end, int_absolute()));
}
float infinity_norm(const thrust::device_ptr<float> &x, int N){
	thrust::device_ptr<float> x_end = x + (N);
	return max( float(thrust::max_element(x, x_end)[0]),
				 abs(float(thrust::min_element(x, x_end)[0]))
				);
}

int main(int argc, char** argv) {
	
	int devNum;
    CUDA_CALL(cudaGetDevice(&devNum));
	printf("Code is executing on device %d \n",devNum );

	// Parse command line arguments
	if (argc < 3) {
		printf("Usage: %s <input_file_path> <output_file_path> [iter_rounds]\n", argv[0]);
		return EXIT_FAILURE;
	}

	std::string file_path = argv[1];
	int iter_rounds = (argc > 3) ? atoi(argv[3]) : 10;
	printf("iter_rounds set to: %d\n", iter_rounds);

	// device copies 
	int *d_row_ptr, *d_col_off, *d_values, *d_b_sum, *d_L; 
	int *d_queue, *d_outbox, *d_outbox_index, *d_outbox_count, *d_cnt, *d_stable_cnt, *d_b_sink_index, *d_sum_Q ;
	float  *d_eta,  *d_eta_max, *d_eta_sum;
	float *d_b, *d_J, *d_b_norm, *d_b_sink,  *d_x, *d_beta;
	curandState *d_state;

	int NE[2];
	printf(".");

	read_file_by_line(file_path, NE, 0, 2);
	printf(".");
	const unsigned int N = NE[0];
	const unsigned int NN = NE[0]+32;
	const unsigned int E = NE[1]; //BECAUSE IN UNDIRECTED GRAPH EVERY EDGE IS COUNTED TWICE
	

	// Alloc space for device copies of graph N E and beta epsilon and kappa eta max
	const unsigned int int_size = sizeof(int);
	const unsigned int float_size = sizeof(float);
	

	CUDA_CALL(cudaMalloc((void **)&d_row_ptr, (NN+1)*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_b, NN*float_size));
	CUDA_CALL(cudaMalloc((void **)&d_J, NN*float_size));  
	CUDA_CALL(cudaMalloc((void **)&d_b_norm, float_size));
	CUDA_CALL(cudaMalloc((void **)&d_x, NN*float_size));

	CUDA_CALL(cudaMalloc((void **)&d_eta_sum, float_size));
	CUDA_CALL(cudaMalloc((void **)&d_eta_max, float_size));
	CUDA_CALL(cudaMalloc((void **)&d_eta, NN*float_size));
	CUDA_CALL(cudaMalloc((void **)&d_b_sink, float_size));
	CUDA_CALL(cudaMalloc((void **)&d_b_sink_index, int_size));
	CUDA_CALL(cudaMalloc((void **)&d_sum_Q, int_size));
	CUDA_CALL(cudaMalloc((void **)&d_beta, float_size));

	CUDA_CALL(cudaMalloc((void **)&d_col_off, E*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_values, E*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_b_sum, int_size));

	CUDA_CALL(cudaMalloc((void **)&d_queue, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_outbox, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_outbox_index, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_outbox_count, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_cnt, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_stable_cnt, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_L, NN*NN*int_size));

	CUDA_CALL(cudaMalloc((void **)&d_state, NN*sizeof(curandState)));
	

	//read the graph and b from input file

	int *row_ptr = (int*)malloc((NN+1)*int_size);
	read_file_by_line(file_path, row_ptr, 1, NN+1);
	printf(".");
	int *col_off = (int*)malloc(E*int_size);
	read_file_by_line(file_path, col_off, 2, E);
	printf(".");
	int *values = (int*)malloc(E*int_size);
	read_file_by_line(file_path, values, 3, E);
	printf(".");
	float *b = (float*)malloc(NN*float_size);
	read_file_by_line(file_path, b, 4, NN);
	printf(".");

	// Copy graph to device
	CUDA_CALL(cudaMemcpy(d_row_ptr, row_ptr, (NN+1)*int_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_col_off, col_off, E*int_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_values, values, E*int_size, cudaMemcpyHostToDevice));	
	CUDA_CALL(cudaMemcpy(d_b, b, NN*float_size, cudaMemcpyHostToDevice));
 
	printf("input and copy to device done \n");

	thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_b);
	thrust::device_ptr<int> int_dev_ptr = thrust::device_pointer_cast(d_queue);
	
	thrust::device_ptr<int> int_dev_ptr2 = thrust::device_pointer_cast(d_queue);
	thrust::device_vector<int> output_keys(N);
	thrust::device_vector<int> output_freqs(N);
	thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;		
	thrust::device_vector<int> dev_ones(N);

	// Host space allocation
	float *eta = (float*)malloc(N*float_size);
	int *queue = (int*)malloc(N*int_size);
	float *rhs_norm = (float*)malloc(float_size);
	float *beta = (float*)malloc(float_size);
	float *result = (float*)malloc(N*sizeof(float));
	float *eta_sum = (float*)malloc(float_size);
	float *eta_max = (float*)malloc(float_size);
	
	// Initial_setup
	double eta_max_threshold = 0.60;
	float frac_of_packet_sunk_threshold = 0.80;

	int num_of_blocks = (N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;  
	unsigned int max_epoch = N; 
	unsigned int epoch;
	int sink_index;
	int Q_sink_index;
	int sum_Q;
	float frac_of_packet_sunk;
	int eta_gt_chk_more_thn_i;
	int frac_of_packet_sunk_more_thn_i;
	float send_recv_rounds;
	bool flag_frac_of_packet = false;

	*beta = 0.0625;
	CUDA_CALL(cudaMemcpy(d_beta, beta, float_size, cudaMemcpyHostToDevice));
	
	get_b_sink<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b, d_b_sink, N, d_b_sink_index);
	CUDA_CALL(cudaMemcpy(&sink_index, d_b_sink_index, int_size, cudaMemcpyDeviceToHost));
	printf("sink index : %d \n", sink_index );

	convert_b_to_J<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b, N, d_b_sink);
	convert_J_to_2betaJ<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b, N, d_beta);
	init_rand<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_state, rand());
				
	CUDA_CALL(cudaMemcpy(b, d_b, N*float_size, cudaMemcpyDeviceToHost));
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	do {
		
		send_recv_rounds = *beta;
		eta_gt_chk_more_thn_i = 0;
		frac_of_packet_sunk_more_thn_i = 0;
		epoch = 0;

		while (send_recv_rounds < 10.0)
			send_recv_rounds *= 10.0;
				
		CUDA_CALL(cudaMemcpy(d_beta, beta, sizeof(float), cudaMemcpyHostToDevice));
		update_b<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b);

		initialize<<<num_of_blocks,THREADS_PER_BLOCK>>>(b, d_eta, d_cnt, d_queue, d_outbox, d_L, sink_index, E, d_state, 2*rand());
		get_b_sink<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b, d_b_sink, N, d_b_sink_index);
		CUDA_CALL(cudaMemcpy(beta, d_b_sink, sizeof(float), cudaMemcpyDeviceToHost));

		*beta = -*beta;
		do {
			for(int i=0; i<(int)send_recv_rounds; i++){
				epoch++;
				reset_outbox<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_outbox);
				send<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_row_ptr, d_b, d_col_off, d_values, d_queue, d_outbox, d_cnt, d_state, rand(), rand(), E);

				CUDA_CALL(cudaDeviceSynchronize());

				thrust::device_ptr<int> outbox_dev_ptr;
				outbox_dev_ptr = thrust::device_pointer_cast(d_outbox);
				thrust::fill(dev_ones.begin(), dev_ones.end(), 1);
				thrust::stable_sort(outbox_dev_ptr, outbox_dev_ptr + N);
				thrust::fill(output_keys.begin(), output_keys.end(), -1);
				thrust::fill(output_freqs.begin(), output_freqs.end(), -1);
				new_end = thrust::reduce_by_key(outbox_dev_ptr, outbox_dev_ptr+N, dev_ones.begin(), output_keys.begin(), output_freqs.begin());

				int_dev_ptr = output_keys.data();
				int_dev_ptr2 = output_freqs.data();

				d_outbox_count = thrust::raw_pointer_cast(int_dev_ptr2);
				d_outbox_index = thrust::raw_pointer_cast(int_dev_ptr);				

				thrust_recv<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_outbox_count, d_queue, d_outbox_index, N);
			
				CUDA_CALL(cudaDeviceSynchronize());
				calculate_eta<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_eta, d_cnt, float(epoch));	
			}

			/**************Termination condition prep if queues are saturated *******************/
			CUDA_CALL(cudaDeviceSynchronize());

			dev_ptr = thrust::device_pointer_cast(d_eta);
			*eta_max = infinity_norm(dev_ptr, N);
			
			printf("epoch: %d \t eta_max_inner: %f\n", epoch, *eta_max);

			/**************Termination condition prep for Q[sink]/(1+sum(Q)) *******************/
			CUDA_CALL(cudaMemcpy(&Q_sink_index, d_queue+sink_index, int_size, cudaMemcpyDeviceToHost));
			int_dev_ptr = thrust::device_pointer_cast(d_queue);
			sum_Q = int_one_norm(int_dev_ptr, N);
			frac_of_packet_sunk = (float)Q_sink_index/(float)(1+sum_Q);
			
			printf(" Frac of packet sunk: %f\t sink: %d\t sum: %d\n", frac_of_packet_sunk, Q_sink_index, sum_Q);

			// Termination in action
			if((*eta_max >= eta_max_threshold)){
				eta_gt_chk_more_thn_i++;
				if (eta_gt_chk_more_thn_i >= 3){
					printf("eta_max is gt threshold so breaking\n");
					break;
				}
			} else if(frac_of_packet_sunk > frac_of_packet_sunk_threshold){
				eta_gt_chk_more_thn_i = 0;
				frac_of_packet_sunk_more_thn_i++;
				if (flag_frac_of_packet != true){
					if (frac_of_packet_sunk_more_thn_i >= 5){ 
						flag_frac_of_packet = true;
						CUDA_CALL(cudaDeviceSynchronize());
						printf("frac_of_packet_sunk is gt threshold so breaking sink: %d\t sum: %d\t frac: %f\n", Q_sink_index, sum_Q, frac_of_packet_sunk);						
						break; 
					}
				}
			} else {
				eta_gt_chk_more_thn_i = 0;
				frac_of_packet_sunk_more_thn_i = 0;
			}

		} while(epoch < max_epoch);

		
	} while((*eta_max) > eta_max_threshold && (*eta_max)>0);

	// =========================================================================
	// ADDED: Iterative NDCG Refinement Post-Loop Block
	// =========================================================================

	// 1 & 2. Get initial eta to host, extract initial top R indices and values
	int r_val = 50;
	if (r_val > (int)N) r_val = N;

	int *d_top_indices, *d_top_values;
	CUDA_CALL(cudaMalloc((void**)&d_top_indices, r_val * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_top_values, r_val * sizeof(int)));

	std::vector<int> h_top_indices(r_val);
	std::vector<int> h_top_values(r_val);
	std::vector<int> prev_idx(r_val), curr_idx(r_val);
	std::vector<float> prev_eta(r_val), curr_eta(r_val);

	// Initial top-R extraction from baseline state
	extractTopRKernel<<<1, 1>>>(d_cnt, N, r_val, d_top_indices, d_top_values);
	CUDA_CALL(cudaDeviceSynchronize());

	CUDA_CALL(cudaMemcpy(h_top_indices.data(), d_top_indices, r_val * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(h_top_values.data(), d_top_values, r_val * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < r_val; i++) {
		prev_idx[i] = h_top_indices[i];
		prev_eta[i] = static_cast<float>(h_top_values[i]);
	}

	// 8. Iterative refinement loop for up to 100 rounds
	int max_rounds = 10000;
	int round_num = 1;
	float ndcg_score = 0.0f;

	printf("\n=======================================================\n");
	printf("Starting Iterative Refinement Loop (iter_rounds = %d)\n", iter_rounds);
	printf("=======================================================\n");

	while (round_num <= max_rounds) {
		// 4. Send/recv rounds that update d_cnt for iter_rounds times
		for (int s = 0; s < iter_rounds; s++) {
			epoch++;
			reset_outbox<<<num_of_blocks, THREADS_PER_BLOCK>>>(d_outbox);
			send<<<num_of_blocks, THREADS_PER_BLOCK>>>(d_row_ptr, d_b, d_col_off, d_values, d_queue, d_outbox, d_cnt, d_state, rand(), rand(), E);

			CUDA_CALL(cudaDeviceSynchronize());

			thrust::device_ptr<int> outbox_dev_ptr = thrust::device_pointer_cast(d_outbox);
			thrust::fill(dev_ones.begin(), dev_ones.end(), 1);
			thrust::stable_sort(outbox_dev_ptr, outbox_dev_ptr + N);
			thrust::fill(output_keys.begin(), output_keys.end(), -1);
			thrust::fill(output_freqs.begin(), output_freqs.end(), -1);
			new_end = thrust::reduce_by_key(outbox_dev_ptr, outbox_dev_ptr + N, dev_ones.begin(), output_keys.begin(), output_freqs.begin());

			int_dev_ptr = output_keys.data();
			int_dev_ptr2 = output_freqs.data();

			d_outbox_count = thrust::raw_pointer_cast(int_dev_ptr2);
			d_outbox_index = thrust::raw_pointer_cast(int_dev_ptr);

			thrust_recv<<<num_of_blocks, THREADS_PER_BLOCK>>>(d_outbox_count, d_queue, d_outbox_index, N);
			CUDA_CALL(cudaDeviceSynchronize());

			calculate_eta<<<num_of_blocks, THREADS_PER_BLOCK>>>(d_eta, d_cnt, float(epoch));
		}

		// 5. Get d_eta to host memory as eta_current
		CUDA_CALL(cudaMemcpy(eta, d_eta, N * sizeof(float), cudaMemcpyDeviceToHost));

		// 6. Extract top indices and values for eta_current
		extractTopRKernel<<<1, 1>>>(d_cnt, N, r_val, d_top_indices, d_top_values);
		CUDA_CALL(cudaDeviceSynchronize());

		CUDA_CALL(cudaMemcpy(h_top_indices.data(), d_top_indices, r_val * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_top_values.data(), d_top_values, r_val * sizeof(int), cudaMemcpyDeviceToHost));

		for (int i = 0; i < r_val; i++) {
			curr_idx[i] = h_top_indices[i];
			curr_eta[i] = static_cast<float>(h_top_values[i]);
		}

		// 7. Calculate NDCG
		ndcg_score = evaluateRoundMetrics(round_num, r_val, curr_idx, curr_eta, prev_idx, prev_eta);

		// 8. Convergence check (< 0.95 continue, >= 0.95 stop)
		if (ndcg_score >= 0.95f) {
			printf("Target NDCG threshold reached (%.5f >= 0.95) at round %d.\n", ndcg_score, round_num);
			break;
		}

		// Shift current round to previous round for next iteration
		prev_idx = curr_idx;
		prev_eta = curr_eta;

		round_num++;
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			
		printf("Epoch: %d \t eta_max: %f \t max_allowed_eta: %f  \t beta: %.10e \t  sink: %d\t sum: %d\t frac: %f \t duartion: ", epoch, *eta_max, eta_max_threshold, *beta, Q_sink_index, sum_Q, frac_of_packet_sunk);
		std::cout<<time_span.count()<<" sec"<<std::endl;
					
		CUDA_CALL(cudaMemcpy(eta, d_eta, (N)*sizeof(float), cudaMemcpyDeviceToHost));

		// solve<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_row_ptr, d_b_sink, d_eta, d_beta, 0.0001 , d_x );
		// CUDA_CALL(cudaMemcpy(eta, d_x, (N)*sizeof(float), cudaMemcpyDeviceToHost));
		
		std::ofstream myfile (argv[2]);
		if (myfile.is_open())
		{
			for(int count = 0; count < N; count ++){
				myfile << eta[count] << "\n" ;
			}
			myfile.close();
		}
		else std::cout << "Unable to open file";


	// Clean up top-R CUDA allocations
	cudaFree(d_top_indices);
	cudaFree(d_top_values);

	// Cleanup
	free(row_ptr);    free(b);    free(eta);    free(col_off);    free(values);    free(rhs_norm);	free(beta);    free(result); 	free(eta_sum); 	  free(eta_max); 
	cudaFree(d_row_ptr); 	cudaFree(d_b); 	cudaFree(d_J); 	cudaFree(d_b_norm); 	cudaFree(d_x); 		cudaFree(d_eta_sum);	cudaFree(d_eta_max); 	
	cudaFree(d_eta); 	cudaFree(d_b_sink); 	cudaFree(d_b_sink_index);	cudaFree(d_beta);	cudaFree(d_col_off); 	cudaFree(d_values); 	cudaFree(d_b_sum);	cudaFree(d_queue); 	cudaFree(d_outbox);	cudaFree(d_cnt);	cudaFree(d_L);	cudaFree(d_state);

	return 0;
}