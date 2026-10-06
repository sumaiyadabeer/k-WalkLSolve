// src/main.cu
#include "../include/config.h"
#include "../include/modules/SolverM.cuh"
#include "../../include/modules/helper.cuh" // Relative path to its header
#include "../include/modules/kwalk.cuh"
#include <iostream>
#include <vector>
#include <numeric> // For std::iota
#include <iomanip> // For std::setprecision
#include <limits>  // For std::numeric_limits


int main() {

    std::cout << "Starting main application..." << std::endl;
    std::cout << "CUDA Sparse Matrix Operations Demo" << std::endl;
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
	printf("Code is executing on device %d \n",GPU_COUNT );

//     if (SOLVER_COUNT == 2 || SOLVER_COUNT == 0){
        
//     int n; // Matrix size will be read from file
//     const int repeat_t = REPEAT_MULT;    // Number of times to repeat multiplication
//     // 1. Take input in CSR format from a file
//     std::string Pfilename = INPUTpmAT;
//     std::string Gfilename = INPUTgRAPH;
//     // std::cout << "\nEnter the input file name (e.g., matrix_input.txt): ";
//     // std::cin >> filename;

//     std::vector<int> row_ptrG;
//     std::vector<int> col_idxG;
//     std::vector<float> valuesG;
//     std::vector<float> b;



//     if (!readGMatrixFromCSRFile(Gfilename, n, row_ptrG, col_idxG, valuesG, b)) {
//         return -1; // Exit if file reading fails
//     }

//     std::vector<int> row_ptr;
//     std::vector<int> col_idx;
//     std::vector<float> values;
//     std::vector<float> start_dis;
//     std::vector<float> final_dis;
//     int sink_index;
//     float sink_value;

//     if (!readPMatrixFromCSRFile(Pfilename, n, row_ptr, col_idx, values, start_dis, sink_index, sink_value )) {
//         return -1; // Exit if file reading fails
//     }
//     std::vector<float> degree = start_dis;
//     std::cout<< sink_index <<" "<< sink_value << std::endl;

//     // // Create an initial input vector (e.g., all ones) for repeated multiplication
//     // std::vector<float> initial_x(n, 1.0f);
//     float sum = std::accumulate(start_dis.begin(), start_dis.end(), 0.0f);
//     if (sum == 0.0f) {
//         std::cout << "Sum of vector elements is zero. Cannot divide by zero. Vector remains unchanged." << std::endl;
//         return;
//     }
//     for (float& element : start_dis) {
//         element /= sum;
//     }
//     // std::cout << "\nInitial input vector (first 10 elements):" << std::endl;
//     // for (int i = 0; i < std::min(10, n); ++i) {
//     //     std::cout << "initial_x[" << i << "] = " << start_dis[i] << std::endl;
//     // }
//     // Create sparse matrix object
//     SparseMatrix sparse_matrix(n, row_ptr, col_idx, values);
//     sparse_matrix.printInfo();
    
//     // 2. Print matrix (CSR components)
//     // For large matrices, printing all CSR components might be too much output.
//     // Adjust this condition based on your expected matrix sizes.
//     if (n <= 10) { 
//         sparse_matrix.printMatrixCSR();
//     } else {
//         std::cout << "\nMatrix is large (n=" << n << "), skipping full CSR component printout." << std::endl;
//     }

//     // Using cuSPARSE for repeated multiplication
//     while(1){
//         final_dis = repeatedMatrixVectorMultiplication(sparse_matrix, start_dis, degree,  repeat_t, true);
//         // std::cout << "resultant  vector (first 10 elements):" << std::endl;
//         // for (int i = 0; i < std::min(10, n); ++i) {
//         //     std::cout << "eta[" << i << "] = " << final_dis[i] << std::endl;
//         // }
//         // Re-create matrix for custom kernel repeated multiplication to ensure fair comparison
//         // (as the previous scalar multiplication modified the matrix values)
//         // std::cout << "\nRe-creating matrix with original values for Custom Kernel repeated multiplication." << std::endl;
//         // // Re-read from file to get original values for a clean comparison
//         // if (!readPMatrixFromCSRFile(Pfilename, n, row_ptr, col_idx, values,start_dis, sink_index, sink_value)) {
//         //     return -1; 
//         // }
//         // for (float& element : start_dis) {
//         //     element /= sum;
//         // }
//         // SparseMatrix sparse_matrix_reinit(n, row_ptr, col_idx, values); // Original matrix values
//         // final_dis = repeatedMatrixVectorMultiplication(sparse_matrix_reinit, start_dis, degree, sink_index, repeat_t, false);
//         std::cout << "resultant  vector (first 10 elements):" << std::endl;
//         for (int i = 0; i < std::min(10, n); ++i) {
//             std::cout << "x[" << i << "] = " << final_dis[i] << std::endl;
//         }
        
//         start_dis = final_dis;
//         float rel_res;
//         rel_res = calculateLaplacianError(sink_index, n, final_dis, row_ptrG, col_idxG, valuesG, b, false);
//         printf("Relative residual in MM is: %f\n", rel_res );

//         if (rel_res <= RESIDUAL_THRESHOLD){
//             printf("Residual threshold reached\n");
//             break;
//         }else{
//             printf("Coninuing the loop for %d more iterations\n", REPEAT_MULT);
//         }
//     }    
//     // auto cen_startM = std::chrono::high_resolution_clock::now();
//     // start_dis = calculateAbsoluteDifferenceSum(n, final_dis, row_ptrG, col_idxG);
//     // auto cen_endM = std::chrono::high_resolution_clock::now();
//     // std::chrono::duration<double> elapsed_M = cen_endM - cen_startM;
//     // printf("M Centarlity time: %ld μs\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed_M).count());


//     // std::cout << "resultant  vector (first 10 elements):" << std::endl;
//     // for (int i = 0; i < std::min(10, n); ++i) {
//     //     std::cout << "centrality[" << i << "] = " << start_dis[i] << std::endl;
//     // }
//     std::ofstream outputFile(OpFileSolM);
//     // Check if the file was opened successfully
//     if (!outputFile.is_open()) {
//         std::cerr << "Error: Unable to open the file." << std::endl;
//         return 1; // Indicate an error
//     }
//     std::cout << "Printing to file" << std::endl;
//     for (int i = 0; i < n; ++i) {
//         outputFile <<  start_dis[i] <<"\t" << b[i] << std::endl;
//     }
//     std::cout << "Printing to file complete" << std::endl;
//     // Close the file
//     outputFile.close();

// }

// ---------------------------------K Walk code starts from here -------------------------------------------
    printf("K-WalkSolver starting: \n");
    // const char* output_path = OpFileSolK;
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

    // printf("Reading graph from file starts: \n");
    int V = read_csr(file_path, &h_row_ptr, &h_col_offset, &h_values, &h_b);
    // printf("Reading graph from file ends: \n");
    //     for (int i = 0; i < std::min(10, V); ++i) {
    //     std::cout<<"h_row_ptr["<<i<< "] = "<< h_row_ptr[i] <<" h_col_offset["<<i<< "] = "<< h_col_offset[i]  <<" h_values["<<i<< "] = "<< h_values[i]  <<" h_b["<<i<< "] = "<< h_b[i]<<std::endl;
    // }
    // std::cout<< V<< "<-V and no of msj -> " << NO_OF_MSJ << std::endl;
    // std::cout<<V<<"  "<<std::log(V)<<std::endl;
    long long max_epoch = ceil(std::log(V)*std::log(V)*std::log(V)*std::log(V)); //(long long)V*(NO_OF_MSJ/100); //MAX_EPOCH;
    std::cout<<"max epoch is: "<< max_epoch<< std::endl;
   
    h_cnt = (int*)calloc(V,sizeof(int));
    // h_queue_size = (int*)calloc(V, sizeof(int));

    int msg_alloc = 0;
    bool* h_recv = (bool*)calloc(V,sizeof(bool));

    for (int i = 0; i < V; i++) {
        if (h_b[i] <= 0) continue;
        int num_msges = round(h_b[i] * num_msg_packets);
        // printf(" %f %d |",b[i],num_msges);
        int end = msg_alloc + num_msges;
        for (int curr = msg_alloc; curr < end  && curr < num_msg_packets; curr++) {
            h_messages[curr].msg_idx = curr;
            h_messages[curr].node = i;
            h_recv[i] = 1;
            h_traveller[curr] = i;
            // h_queue_size[i]++;
            msg_alloc++;
        }
    }
    for(int i = 0 ; i<V ; i++){
        if(h_recv[i]){
            h_cnt[i]++;
        }
    }
    // printf("B2J conversion start\n");
    b2J(V, h_b, h_negValue, h_negIndex, source_nodes, sink_nodes);
    h_source_idx = (int*)malloc(sizeof(int) * source_nodes);
    int w = 0;
    for(int i = 0 ; i < V ;i++){
        if(h_b[i] > 0){
            h_source_idx[w] = i;
            w++;
        }
    }
    
    // for(int i = 0 ; i<source_nodes ; i++){
    // 	printf("Source index: %d\n",h_source_idx[i]);
    // }

    printf("Number of packets in network : %d\n",msg_alloc);
    printf("Negative index : %d \t %f\n", *h_negIndex, h_b[*h_negIndex]);
    if (V == -1) {
        return -1;
    }
    float *h_eta = (float*)calloc(V,sizeof(float));
    int *h_epsilon = (int*)calloc(V,sizeof(int));
    float* h_prev_eta = (float*)calloc(V, sizeof(float));


    // printf("upto here done bf variable declare \n");

    int *d_row_ptr, *d_col_offset, *d_queue_sizes, *d_cnt,*del_d_cnt, *d_source_idx, *d_traveller;
    float *d_b, *d_eta, *d_bc, *d_prev_eta, *d_values;
    bool *d_recv;
    double* d_err;
    struct message *d_messages;
    curandState *d_state;
    // printf("upto here done bf variable mem alloc \n");
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_source_idx, sizeof(int)* source_nodes));
    CHECK_CUDA(cudaMalloc(&d_row_ptr, sizeof(int) * (V + 1)));
    CHECK_CUDA(cudaMalloc(&d_col_offset, sizeof(int) * h_row_ptr[V]));
    CHECK_CUDA(cudaMalloc(&d_values, sizeof(int) * h_row_ptr[V]));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(float) * V));
    CHECK_CUDA(cudaMalloc(&d_queue_sizes, sizeof(int) * V));
    CHECK_CUDA(cudaMalloc(&d_cnt, sizeof(int) * V));
    CHECK_CUDA(cudaMalloc(&del_d_cnt, sizeof(int) * V));
    CHECK_CUDA(cudaMalloc(&d_eta, sizeof(float) * V));
    CHECK_CUDA(cudaMalloc(&d_bc, sizeof(float) * V));
    CHECK_CUDA(cudaMalloc(&d_prev_eta, sizeof(float) * V));
    CHECK_CUDA(cudaMalloc(&d_messages, sizeof(struct message) * num_msg_packets));
    CHECK_CUDA(cudaMalloc(&d_state, sizeof(curandState) * num_msg_packets));  // Allocate memory for RNG states
    CHECK_CUDA(cudaMalloc(&d_recv, sizeof(bool) * V));
    CHECK_CUDA(cudaMalloc(&d_err, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_traveller, sizeof(int) * num_msg_packets));
    // cudaMalloc(&d_queue_size, sizeof(int) * V);

    // printf("upto here done bf variable set 0 \n");
    cudaMemset(d_recv, 0, sizeof(bool) * V);
    cudaMemset(d_cnt, 0, sizeof(int) * V);
    cudaMemset(del_d_cnt, 0, sizeof(int) * V);
    cudaMemset(d_eta, 0, sizeof(float) * V);
    cudaMemset(d_bc, 0, sizeof(float) * V);
    cudaMemset(d_prev_eta, 0, sizeof(float) * V);
    cudaMemset(d_traveller, 0, sizeof(int) * num_msg_packets);

    // printf("upto here done bf variable copy frm host \n");
    // Copy data from host to device
    cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_offset, h_col_offset, sizeof(int) * h_row_ptr[V], cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, sizeof(float) * h_row_ptr[V], cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_messages, h_messages, sizeof(struct message) * num_msg_packets, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, h_cnt, sizeof(int) * V, cudaMemcpyHostToDevice);
    // cudaMemcpy(del_d_cnt, h_cnt, sizeof(int) * V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_idx, h_source_idx, sizeof(int) * source_nodes, cudaMemcpyHostToDevice);

    // printf("upto here done bf kernel launch \n");
    int blockSize = 256;
    int msgBlocks = (num_msg_packets + blockSize - 1) / blockSize;
    int nodeBlocks = (V + blockSize - 1) / blockSize;

    
    // Setup the RNG states
    setup_kernel<<<msgBlocks, blockSize>>>(d_state, time(NULL));
    cudaDeviceSynchronize();
    
    flush_boolArray<<<nodeBlocks, blockSize>>>(d_recv, V);
    cudaDeviceSynchronize();
    
    while(1){
    auto start = std::chrono::high_resolution_clock::now();
    int epoch = 1;
    std::cout<<"Max epoch: "<< max_epoch<<std::endl;
    while (epoch < max_epoch){
        // epoch = epoch + num_msg_packets;
        // printf("Epoch: %d", epoch);
        if(epoch%100000==0){
            std::cout<<"Epoch: "<< epoch << "Max_epoch: " << max_epoch <<std::endl;}
        processMessages<<<msgBlocks, blockSize>>>(d_row_ptr, d_col_offset, d_values, d_messages, num_msg_packets, V,d_b ,d_source_idx,
                                                  source_nodes ,d_state, d_recv, *h_negIndex, d_traveller);
        cudaDeviceSynchronize();
        // set_qlen2zero<<<nodeBlocks, blockSize>>>(d_cnt, V);
        // cudaDeviceSynchronize();

        // cudaMemcpyDeviceToDevice
        cudaMemcpy(del_d_cnt, d_cnt, sizeof(int) * V, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        update_qlen<<<msgBlocks, blockSize>>>(d_traveller, num_msg_packets, d_cnt);
        cudaDeviceSynchronize();
        cnt_diff<<<msgBlocks, blockSize>>>(d_cnt, del_d_cnt, V);
        cudaDeviceSynchronize();
        update_eta_by_qLen<<<nodeBlocks, blockSize>>>(d_eta, d_prev_eta, d_cnt, V, d_recv, epoch, num_msg_packets);
        cudaDeviceSynchronize();
        epoch++;
    }
    // Calculate norm(del_d_cnt)/t and report as epsilon
    cudaMemcpy(h_epsilon, del_d_cnt, sizeof(float) * V, cudaMemcpyDeviceToHost);
   
    // printf("Exiting the inner loop\n");
    // convert eta to x by diving with degree
    // eta2x<<<nodeBlocks, blockSize>>>(d_eta, V, msg_alloc, d_row_ptr);
    // cudaDeviceSynchronize();
    // groundind the sink vertex
    cudaMemcpy(h_eta, d_eta, sizeof(float) * V, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // h_eta[*h_negIndex]=0;
    // cudaDeviceSynchronize();
    // cudaMemcpy(d_eta, h_eta, sizeof(float) * V, cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();
    for (int i = 0; i < std::min(10, V); ++i) {
        std::cout<<"x["<<i<< "] = "<< h_eta[i]/NO_OF_MSJ <<std::endl;
    }
    auto endd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endd - start;

    printf("K Elapsed time: %ld μs\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

    float rel_res_k;
    h_b[*h_negIndex] = -1;
    rel_res_k = Wrapper_calculateLaplacianError(*h_negIndex, V, h_row_ptr[V], h_eta,
        h_row_ptr, h_col_offset, h_b, 
        false);
    printf("Realtive Residual in Kwalk: %f\n", rel_res_k);



    // auto cen_startK = std::chrono::high_resolution_clock::now();
    // calculate_abs_diff_sum_kernel<<<nodeBlocks, blockSize>>>(V,  d_row_ptr, d_col_offset,  d_eta,  d_bc);
    // // normalise<<<nodeBlocks, blockSize>>>(d_eta, V, max_epoch);
    // auto cen_endK = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_K = cen_endK - cen_startK;

    // printf("K Centralty time: %ld μs\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed_K).count());

    // // Copy results back to host if needed
    // cudaMemcpy(h_eta, d_bc, sizeof(float) * V, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_prev_eta, d_prev_eta, sizeof(float) * V, cudaMemcpyDeviceToHost);
    float err = 0;
    for(int i  =  1 ; i < V ; i++){
        if(h_prev_eta[i]!=0){
            err = max(err, abs(h_eta[i] - h_prev_eta[i])/h_prev_eta[i]);
            // printf("Eta and Prev_eta : %f \t %f : ",h_eta[i], h_prev_eta[i]);
        }
    }
    printf("Max eta_delta = %f\n", err);
    // FILE* eta_graph = fopen(output_path,"w");


    printf("#Vertices: %d\n#Source nodes: %d\n#Sink nodes: %d\n#Message Packets: %d\nError: %f\nTime taken: %ld ms\n", V, source_nodes,sink_nodes,num_msg_packets, err, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

    if (rel_res_k <= RESIDUAL_THRESHOLD || !RESIDUAL_RUN){
        printf("Residual threshold reached\n");
        break;
    }else{
        printf("Coninuing the loop for %d more epoch\n", MAX_EPOCH);
    }
}

    // printf("Eta distribution: \n");
    std::ofstream outputFile(OpFileSolK);
    // Check if the file was opened successfully
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open the file." << std::endl;
        return 1; // Indicate an error
    }
    // for (int i = 0; i < std::min(10, V); ++i) {
    //     std::cout<< "centrality["<<i<< "] = "<< h_eta[i] <<std::endl;
    // }
    std::cout << "Printing to file OpFileSolK" << std::endl;
    for (int i = 0; i < V; ++i) {
        // std::cout<< h_eta[i] <<std::endl;
        outputFile <<  h_eta[i]/NO_OF_MSJ <<  "\n"; //h_b[i]?1:0 
    }
    std::cout << "Printing to file OpFileSolK complete" << std::endl;
    // Close the file
    outputFile.close();

    cudaFree(d_row_ptr);
    cudaFree(d_col_offset);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_queue_sizes);
    cudaFree(d_cnt);
    cudaFree(del_d_cnt);
    cudaFree(d_eta);
    cudaFree(d_bc);
    cudaFree(d_prev_eta);
    cudaFree(d_messages);
    cudaFree(d_state);
    cudaFree(d_recv);
    cudaFree(d_err);

   
    free(h_row_ptr);
    free(h_col_offset);
    free(h_values);
    free(h_b);
    free(h_negIndex);
    free(h_cnt);
    free(h_negValue);
    free(h_eta);
    free(h_prev_eta);
    free(h_recv);
    free(h_messages);
    
 
    std::cout << "\nMain application finished." << std::endl;
    std::cout << "\n----------------------------------------\n";
    std::cout << "Program finished. Press ENTER to exit...";
    std::cin.get(); // Waits for user input before exiting
    return 0;
}
