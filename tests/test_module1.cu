// tests/test_module1.cu
#include "../include/modules/module1.cuh"
#include "../include/config.h"
#include <iostream>
#include <vector>
#include <numeric> // For std::iota
#include <cmath>   // For std::fabs

int main() {
    std::cout << "Running unit test for Module1 (Array Addition)..." << std::endl;

    int size = DEFAULT_ARRAY_SIZE;
    std::vector<float> h_a(size);
    std::vector<float> h_b(size);
    std::vector<float> h_c_actual(size); // Result from CUDA
    std::vector<float> h_c_expected(size); // Expected result

    std::iota(h_a.begin(), h_a.end(), 1.0f);
    std::iota(h_b.begin(), h_b.end(), 10.0f);

    // Calculate expected results on host
    for (int i = 0; i < size; ++i) {
        h_c_expected[i] = h_a[i] + h_b[i];
    }

    // Run the CUDA function
    bool success = runArrayAddition(h_a.data(), h_b.data(), h_c_actual.data(), size);

    if (!success) {
        std::cerr << "Module1 test failed during CUDA execution." << std::endl;
        return 1;
    }

    // Verify results
    bool test_passed = true;
    for (int i = 0; i < size; ++i) {
        if (std::fabs(h_c_actual[i] - h_c_expected[i]) > 1e-5) { // Use a small tolerance for float comparison
            std::cerr << "Mismatch at index " << i << ": Expected " << h_c_expected[i] << ", Got " << h_c_actual[i] << std::endl;
            test_passed = false;
            break;
        }
    }

    if (test_passed) {
        std::cout << "Module1 Unit Test: PASSED" << std::endl;
        return 0;
    } else {
        std::cerr << "Module1 Unit Test: FAILED" << std::endl;
        return 1;
    }
}
