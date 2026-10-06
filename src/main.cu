// src/main.cu
#include "../include/config.h"
#include "../include/modules/SolverM.cuh"
#include "../../include/modules/helper.cuh"
#include "../include/modules/kwalk.cuh"

#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <map>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>


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

double computeRelativeResidual(int V, 
                               const std::vector<float>& h_eta, 
                               const int* h_row_ptr, 
                               const int* h_col_offset, 
                               const float* h_values, 
                               const float* h_b_orig, 
                               int negIndex) {
    // 1. Calculate x vector (x = eta / degree) and RHS vector b
    std::vector<float> x(V, 0.0f);
    std::vector<float> degree(V, 0.0f);
    std::vector<float> b_vec(V);

    for (int i = 0; i < V; ++i) {
        degree[i] = static_cast<float>(h_row_ptr[i + 1] - h_row_ptr[i]);
        x[i] = (degree[i] > 0.0f) ? (h_eta[i] / degree[i]) : 0.0f;
        b_vec[i] = h_b_orig[i];
    }

    // Apply grounding and RHS boundary condition at sink
    if (negIndex >= 0 && negIndex < V) {
        x[negIndex] = 0.0f;
        b_vec[negIndex] = -1.0f;
    }

    // 2. Compute Ax, Dx (which is exact h_eta), Lx, and Residual in a single pass
    std::vector<float> Lx(V, 0.0f);
    std::vector<float> residual(V, 0.0f);
    double norm_res_sq = 0.0;
    double norm_b_sq   = 0.0;

    for (int i = 0; i < V; ++i) {
        int start = h_row_ptr[i];
        int end   = h_row_ptr[i + 1];
        
        float p_x_sum = 0.0f;
        for (int j = start; j < end; ++j) {
            p_x_sum += h_values[j] * x[h_col_offset[j]];
        }
        
        float Ax_i = degree[i] * p_x_sum;
        
        // Exact Dx is h_eta (grounded at sink)
        float Dx_i = (i == negIndex) ? 0.0f : h_eta[i];
        
        Lx[i] = Dx_i - Ax_i;
        residual[i] = Lx[i] - b_vec[i];

        norm_res_sq += static_cast<double>(residual[i]) * static_cast<double>(residual[i]);
        norm_b_sq   += static_cast<double>(b_vec[i]) * static_cast<double>(b_vec[i]);
    }

    // Console Diagnostics
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << " LAPLACIAN SYSTEM DETAILED VALUES (Lx, b, Lx - b) " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    
    for (int i = 0; i < std::min(10, V); ++i) {
        std::cout << "Lx[" << i << "] = " << Lx[i] 
                  << "\t b[" << i << "] = " << b_vec[i] 
                  << "\t Lx - b[" << i << "] = " << residual[i] << std::endl;
    }

    if (negIndex >= 0 && negIndex < V) {
        std::cout << "Lx[" << negIndex << "] = " << Lx[negIndex] 
                  << "\t b[" << negIndex << "] = " << b_vec[negIndex] 
                  << "\t Lx - b[" << negIndex << "] = " << residual[negIndex] << std::endl;
    }
    std::cout << "--------------------------------------------------\n" << std::endl;

    double norm_res = std::sqrt(norm_res_sq);
    double norm_b   = std::sqrt(norm_b_sq);
    double rel_res  = (norm_b > 0.0) ? (norm_res / norm_b) : norm_res;

    std::cout << " Norm ||Lx - b||_2 : " << norm_res << std::endl;
    std::cout << " Norm ||b||_2      : " << norm_b << std::endl;
    std::cout << " Relative Residual : " << rel_res << std::endl;
    std::cout << "--------------------------------------------------\n" << std::endl;

    return rel_res;
}

int main() {
    std::cout << "Starting main application..." << std::endl;
    std::cout << "CUDA Parallel K-Walk Solver Pipeline" << std::endl;
    std::cout << "==================================" << std::endl;
    
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
    
    CHECK_CUDA(cudaSetDevice(GPU_COUNT));
    printf("Code is executing on device %d \n", GPU_COUNT);

    // --------------------------------- K Walk Initial Setup -------------------------------------------
    printf("K-WalkSolver starting: \n");
    const char* file_path = INPUTgRAPH;
    int num_msg_packets = NO_OF_MSJ;
    int *h_row_ptr = NULL, *h_col_offset = NULL, *h_negIndex = NULL, *h_cnt = NULL, *h_source_idx = NULL, *h_traveller = NULL;
    int source_nodes = 0;
    int sink_nodes = 0;
    float *h_values = NULL, *h_b = NULL, *h_negValue = NULL;

    h_negValue = (float*)malloc(sizeof(float));
    h_negIndex = (int*)malloc(sizeof(int));

    struct message *h_messages = (struct message*)malloc(sizeof(struct message) * num_msg_packets);
    h_traveller = (int*)malloc(sizeof(int) * num_msg_packets);

    int V = read_csr(file_path, &h_row_ptr, &h_col_offset, &h_values, &h_b);
    if (V == -1) {
        std::cerr << "Failed to read graph CSR inputs!" << std::endl;
        return -1;
    }

    long long max_epoch = MAX_EPOCH;
    std::cout << "Max epoch per iteration batch: " << max_epoch << std::endl;
   
    h_cnt = (int*)calloc(V, sizeof(int));
    int msg_alloc = 0;
    bool* h_recv = (bool*)calloc(V, sizeof(bool));

    for (int i = 0; i < V; i++) {
        if (h_b[i] <= 0) continue;
        int num_msges = round(h_b[i] * num_msg_packets);
        int end = msg_alloc + num_msges;
        for (int curr = msg_alloc; curr < end && curr < num_msg_packets; curr++) {
            h_messages[curr].msg_idx = curr;
            h_messages[curr].node = i;
            h_recv[i] = 1;
            h_traveller[curr] = i;
            msg_alloc++;
        }
    }
    for (int i = 0; i < V; i++) {
        if (h_recv[i]) {
            h_cnt[i]++;
        }
    }

    b2J(V, h_b, h_negValue, h_negIndex, source_nodes, sink_nodes);
    h_source_idx = (int*)malloc(sizeof(int) * source_nodes);
    int w = 0;
    for (int i = 0; i < V; i++) {
        if (h_b[i] > 0) {
            h_source_idx[w] = i;
            w++;
        }
    }

    printf("Number of packets in network: %d\n", msg_alloc);
    printf("Negative index (Sink): %d \t %f\n", *h_negIndex, h_b[*h_negIndex]);

    // Host Backup Arrays for Re-seeding Multi-Round Workloads
    struct message *h_initial_messages = (struct message*)malloc(sizeof(struct message) * num_msg_packets);
    int *h_initial_traveller = (int*)malloc(sizeof(int) * num_msg_packets);
    memcpy(h_initial_messages, h_messages, sizeof(struct message) * num_msg_packets);
    memcpy(h_initial_traveller, h_traveller, sizeof(int) * num_msg_packets);

    // Device Memory Allocations
    int *d_row_ptr, *d_col_offset, *d_cnt, *del_d_cnt, *d_source_idx, *d_traveller, *d_traveller_2;
    int *d_non_s_cnt, *d_s_cnt;
    float *d_b, *d_eta, *d_values;
    bool *d_recv;
    struct message *d_messages, *d_messages2;
    unsigned long long *d_at_sink, *d_walk_gen;
    unsigned long long *at_sink_host = (unsigned long long*)malloc(sizeof(unsigned long long));

    CHECK_CUDA(cudaMalloc(&d_source_idx, sizeof(int) * source_nodes));
    CHECK_CUDA(cudaMalloc(&d_row_ptr, sizeof(int) * (V + 1)));
    CHECK_CUDA(cudaMalloc(&d_col_offset, sizeof(int) * h_row_ptr[V]));
    CHECK_CUDA(cudaMalloc(&d_values, sizeof(float) * h_row_ptr[V]));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(float) * V));
    CHECK_CUDA(cudaMalloc(&d_cnt, sizeof(int) * V));
    CHECK_CUDA(cudaMalloc(&del_d_cnt, sizeof(int) * V));
    CHECK_CUDA(cudaMalloc(&d_eta, sizeof(float) * V));
    CHECK_CUDA(cudaMalloc(&d_messages, sizeof(struct message) * num_msg_packets));
    CHECK_CUDA(cudaMalloc(&d_messages2, sizeof(struct message) * num_msg_packets));
    CHECK_CUDA(cudaMalloc(&d_recv, sizeof(bool) * V));
    CHECK_CUDA(cudaMalloc(&d_traveller, sizeof(int) * num_msg_packets));
    CHECK_CUDA(cudaMalloc(&d_traveller_2, sizeof(int) * num_msg_packets));
    CHECK_CUDA(cudaMalloc(&d_non_s_cnt, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_s_cnt, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_at_sink, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_walk_gen, sizeof(unsigned long long)));

    // Copy static graph topology structures to Device
    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int) * (V + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_offset, h_col_offset, sizeof(int) * h_row_ptr[V], cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, sizeof(float) * h_row_ptr[V], cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, sizeof(float) * V, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cnt, h_cnt, sizeof(int) * V, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_source_idx, h_source_idx, sizeof(int) * source_nodes, cudaMemcpyHostToDevice));

    // Execution Hyperparameters
    int blockSize = 256;
    int msgBlocks = (num_msg_packets + blockSize - 1) / blockSize;
    int nodeBlocks = (V + blockSize - 1) / blockSize;
    const int batch_size = 500;
    float alpha = 0.5f;
    unsigned long long seed = (unsigned long long)time(NULL);

    // Multi-Round Lifecycle Tracking Maps
    int max_rounds = 5;
    std::vector<int> target_r_values = {50}; //, 100, 150, 200, 500
    std::map<int, std::vector<int>> historical_top_indices;
    std::map<int, std::vector<float>> historical_top_etas;
    std::cout << "\n==================================================" << std::endl;
    std::cout << "STARTING MULTI-ROUND OPTIMIZED K-WALK SOLVER..." << std::endl;
    std::cout << "==================================================\n" << std::endl;
    bool early_exit_triggered = false;
    const float NDCG_THRESHOLD = 0.95f;
    const int CHECK_TARGET_R = 50; // Evaluates NDCG termination against R=500 target
    auto start_time_total = std::chrono::high_resolution_clock::now();
    int actual_rounds_executed;

    for (int round = 0; round < max_rounds; round++) {
        int epoch = 1;
        actual_rounds_executed = round+1;
        int revised_no_of_blocks = msgBlocks;
        int no_of_active_Walks = num_msg_packets;
        int prev_active_walks = 2 * num_msg_packets;
        bool orig_data = true;

        // Reset Registers for the new Round
        CHECK_CUDA(cudaMemset(d_at_sink, 0, sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(d_non_s_cnt, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_s_cnt, 0, sizeof(int)));

        // Re-initialize Walker state arrays back into Primary Buffers
        CHECK_CUDA(cudaMemcpy(d_messages, h_initial_messages, sizeof(struct message) * num_msg_packets, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_messages2, h_initial_messages, sizeof(struct message) * num_msg_packets, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_traveller, h_initial_traveller, sizeof(int) * num_msg_packets, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_traveller_2, h_initial_traveller, sizeof(int) * num_msg_packets, cudaMemcpyHostToDevice));

        std::cout << ">>> Beginning Round " << (round + 1) << " Execution..." << std::endl;

        while (true) {
            // Snapshot counts for error tracking
            cudaMemcpy(del_d_cnt, d_cnt, sizeof(int) * V, cudaMemcpyDeviceToDevice);

            auto start_time_batch_walk = std::chrono::high_resolution_clock::now();

            // Unrolled asynchronous batch execution
            for (int i = 0; i < batch_size; i++) {
                if (orig_data) {
                    processMessages_Optimized<<<revised_no_of_blocks, blockSize>>>(
                        d_row_ptr, d_col_offset, d_values, d_messages,
                        num_msg_packets, V, d_b, d_source_idx, source_nodes,
                        d_traveller, epoch, max_epoch, d_at_sink, d_walk_gen, d_cnt, seed
                    );
                } else {
                    processMessages_Optimized<<<revised_no_of_blocks, blockSize>>>(
                        d_row_ptr, d_col_offset, d_values, d_messages2,
                        num_msg_packets, V, d_b, d_source_idx, source_nodes,
                        d_traveller_2, epoch, max_epoch, d_at_sink, d_walk_gen, d_cnt, seed
                    );
                }
                epoch++;
            }

            // Sync Point
            CHECK_CUDA(cudaMemcpy(at_sink_host, d_at_sink, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            // no_of_active_Walks = num_msg_packets - *at_sink_host;

            // Clamp active walks to prevent unsigned underflow
            long long current_sink_count = static_cast<long long>(*at_sink_host);
            no_of_active_Walks = std::max(0, num_msg_packets - static_cast<int>(current_sink_count));
            // Restructuring Gate: Epoch > max_epoch and population threshold triggers
            if (epoch > max_epoch && (num_msg_packets - *at_sink_host) > blockSize) {
                if (((float)no_of_active_Walks) / ((float)prev_active_walks) < alpha) {
                    auto start_time_awc = std::chrono::high_resolution_clock::now();

                    // CRITICAL RESET: Zero out atomic trackers before running compaction
                    cudaMemset(d_non_s_cnt, 0, sizeof(int));
                    cudaMemset(d_s_cnt, 0, sizeof(int));

                    revised_no_of_blocks = (no_of_active_Walks + blockSize - 1) / blockSize;
                    prev_active_walks = no_of_active_Walks;

                    if (orig_data) {
                        partitionValueToRight2<<<msgBlocks, blockSize, blockSize * sizeof(int)>>>(
                            d_traveller_2, d_traveller, d_messages2, d_messages, 
                            num_msg_packets, -1, d_non_s_cnt, d_s_cnt
                        );
                    } else {
                        partitionValueToRight2<<<msgBlocks, blockSize, blockSize * sizeof(int)>>>(
                            d_traveller, d_traveller_2, d_messages, d_messages2, 
                            num_msg_packets, -1, d_non_s_cnt, d_s_cnt
                        );
                    }
                    orig_data = !orig_data;
                    cudaDeviceSynchronize();

                    auto end_time_awc = std::chrono::high_resolution_clock::now();
                    auto mss = std::chrono::duration_cast<std::chrono::microseconds>(end_time_awc - start_time_awc).count();
                    std::cout << " [AWC] Compacted grid. Time taken: " << mss << " us | Active Walks: " 
                              << no_of_active_Walks << " | Revised Blocks: " << revised_no_of_blocks << std::endl;
                }
            }

            if ((epoch - 1) % 1000 == 0) {
                std::cout << "Epoch: " << epoch << " | Walks at Sink: " << *at_sink_host 
                          << " | Revised Blocks: " << revised_no_of_blocks << std::endl;
            }

            if (*at_sink_host >= (unsigned long long)num_msg_packets) {
                printf("Round %d Terminated | Total Epochs: %d | Final Sink Count: %lld\n", round + 1, epoch, *at_sink_host);
                break;
            }
        }

        // =========================================================================
        // POST-ROUND METRIC EVALUATION
        // =========================================================================
        unsigned long long total_global_walks = 0;
        CHECK_CUDA(cudaMemcpy(&total_global_walks, d_walk_gen, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        if (total_global_walks == 0) {
            total_global_walks = (unsigned long long)num_msg_packets * (round + 1);
        }

        // Reset device counter for next round
        // CHECK_CUDA(cudaMemset(d_walk_gen, 0, sizeof(unsigned long long)));

        for (int r : target_r_values) {
            int* d_top_indices;
            int* d_top_values;
            CHECK_CUDA(cudaMalloc(&d_top_indices, r * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_top_values, r * sizeof(int)));

            extractTopRKernel<<<1, 1>>>(d_cnt, V, r, d_top_indices, d_top_values);
            cudaDeviceSynchronize();

            std::vector<int> h_top_indices(r);
            std::vector<int> h_top_values(r);
            CHECK_CUDA(cudaMemcpy(h_top_indices.data(), d_top_indices, r * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_top_values.data(), d_top_values, r * sizeof(int), cudaMemcpyDeviceToHost));

            std::vector<float> h_top_etas(r);
            for (int i = 0; i < r; i++) {
                h_top_etas[i] = static_cast<float>(h_top_values[i]) / total_global_walks;
            }

            if (round > 0) {
                // Evaluates NDCG only
                float current_ndcg = evaluateRoundMetrics(round + 1, r, 
                                                           h_top_indices, h_top_etas, 
                                                           historical_top_indices[r], historical_top_etas[r]);

                // Termination check based on NDCG threshold
                if (r == CHECK_TARGET_R && current_ndcg >= NDCG_THRESHOLD) {
                    std::cout << "\n[TERMINATION TRIGGER] NDCG (" << current_ndcg 
                              << ") reached threshold " << NDCG_THRESHOLD 
                              << " at Round " << (round + 1) << ". Stopping multi-round loop." << std::endl;
                    early_exit_triggered = true;
                }
            }

            historical_top_indices[r] = h_top_indices;
            historical_top_etas[r] = h_top_etas;

            cudaFree(d_top_indices);
            cudaFree(d_top_values);
        }

        if (early_exit_triggered) {
            break;
        }
    }

    auto end_time_total = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_total - start_time_total).count();
    printf("\nTotal Multi-Round Execution Time: %ld ms\n", total_ms);

    // =========================================================================
    // FINAL POST-PROCESSING: ETA COMPUTATION, PRINTING & FILE SAVE
    // =========================================================================
    unsigned long long total_generated_final = 0;
    CHECK_CUDA(cudaMemcpy(&total_generated_final, d_walk_gen, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Exact total walks across all completed rounds
    unsigned long long final_total_walks = ((unsigned long long)num_msg_packets * actual_rounds_executed) + total_generated_final;
    
        // Compute Eta = d_cnt / final_total_walks on GPU
    update_eta_by_q<<<nodeBlocks, blockSize>>>(d_eta, d_cnt, V, final_total_walks);
    cudaDeviceSynchronize();

    std::vector<float> h_final_eta(V);
    CHECK_CUDA(cudaMemcpy(h_final_eta.data(), d_eta, sizeof(float) * V, cudaMemcpyDeviceToHost));

    // Print first 10 calculated Eta values
    std::cout << "\nFirst 10 calculated Eta (cnt / total_walks) values:" << std::endl;
    for (int i = 0; i < std::min(10, V); ++i) {
        std::cout << "eta[" << i << "] = " << h_final_eta[i] << std::endl;
    }

    // Compute relative residual and print Lx, b, and Lx-b for first 10 & sink vertex
    double relative_residual = computeRelativeResidual(V, h_final_eta, h_row_ptr, h_col_offset, h_values, h_b, *h_negIndex);

    std::cout << "Final Relative Residual calculated: " << relative_residual << std::endl;

    // SAVE SOLUTION FILE WITH ETA (cnt/total number of walks)
    std::ofstream outputFile(OpFileSolK);
    if (outputFile.is_open()) {
        std::cout << "Writing solution output (Eta values) to file: " << OpFileSolK << std::endl;
        for (int i = 0; i < V; ++i) {
            outputFile << h_final_eta[i] << "\n";
        }
        outputFile.close();
        std::cout << "Printing to file OpFileSolK complete." << std::endl;
    } else {
        std::cerr << "Error: Unable to open output file " << OpFileSolK << std::endl;
    }

    // Cleanup Device and Host Memory Allocation
    cudaFree(d_row_ptr); cudaFree(d_col_offset); cudaFree(d_values);
    cudaFree(d_b); cudaFree(d_cnt); cudaFree(del_d_cnt);
    cudaFree(d_eta); cudaFree(d_messages); cudaFree(d_messages2);
    cudaFree(d_recv); cudaFree(d_traveller); cudaFree(d_traveller_2);
    cudaFree(d_source_idx); cudaFree(d_non_s_cnt); cudaFree(d_s_cnt);
    cudaFree(d_at_sink); cudaFree(d_walk_gen);

    free(h_row_ptr); free(h_col_offset); free(h_values); free(h_b);
    free(h_negIndex); free(h_cnt); free(h_negValue); 
    // free(h_eta);
    free(h_recv); free(h_messages); free(h_traveller);
    free(h_initial_messages); free(h_initial_traveller); free(at_sink_host);

    std::cout << "\nMain application finished." << std::endl;
    std::cout << "\n----------------------------------------\n";
    std::cout << "Program finished. Press ENTER to exit...";
    std::cin.get();
    return 0;
}
