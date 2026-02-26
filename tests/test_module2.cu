// tests/test_module2.cu
// #include "../include/modules/module2.cuh"
#include "../include/config.h"
#include <iostream>
#include <vector>
#include <numeric> // For std::iota
#include <cmath>   // For std::fabs

int main() {
    std::cout << "Running unit test for Module2 (Scalar Multiplication)..." << std::endl;

    int size = DEFAULT_ARRAY_SIZE;
    std::vector<float> h_in(size);
    std::vector<float> h_out_actual(size); // Result from CUDA
    std::vector<float> h_out_expected(size); // Expected result
    float scalar_val = 5.5f;

    std::iota(h_in.begin(), h_in.end(), 1.0f); // h_in = {1.0, 2.0, ..., size.0}

    // Calculate expected results on host
    for (int i = 0; i < size; ++i) {
        h_out_expected[i] = h_in[i] * scalar_val;
    }

    // // Run the CUDA function
    // bool success = runScalarMultiplication(h_in.data(), h_out_actual.data(), scalar_val, size);

    // if (!success) {
    //     std::cerr << "Module2 test failed during CUDA execution." << std::endl;
    //     return 1;
    // }

    // // Verify results
    // bool test_passed = true;
    // for (int i = 0; i < size; ++i) {
    //     if (std::fabs(h_out_actual[i] - h_out_expected[i]) > 1e-5) { // Use a small tolerance for float comparison
    //         std::cerr << "Mismatch at index " << i << ": Expected " << h_out_expected[i] << ", Got " << h_out_actual[i] << std::endl;
    //         test_passed = false;
    //         break;
    //     }
    // }

    // if (test_passed) {
    //     std::cout << "Module2 Unit Test: PASSED" << std::endl;
    //     return 0;
    // } else {
    //     std::cerr << "Module2 Unit Test: FAILED" << std::endl;
    //     return 1;
    // }
}
