import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import cg
import time

def solve_laplacian(input_filename, output_filename):
    """
    Solves Lx = b for a graph Laplacian and saves the solution to a file.
    """
    print(f"1. Reading data from '{input_filename}'...")
    try:
        with open(input_filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
        return

    # 2. Parse the input file
    try:
        # First line: no_of_vertices, no_of_edges
        v, e = map(int, lines[0].strip().split())

        # Next lines: row_ptr, col_offset, values, b
        row_ptr = list(map(int, lines[1].strip().split()))
        col_offset = list(map(int, lines[2].strip().split()))
        values = list(map(float, lines[3].strip().split()))
        b = np.array(list(map(float, lines[4].strip().split())))

    except (ValueError, IndexError):
        print("Error: The input file format is incorrect. Please check the data.")
        return

    print(f"Parsed graph with {v} vertices and {e} edges.")

    # 3. Construct the sparse Laplacian matrix (CSR format)
    L = csr_matrix((values, col_offset, row_ptr), shape=(v, v))
    a = csgraph.laplacian(L)
    a = csr_matrix(a)


    sink_index = [i for i in range(0, b.size) if b[i] < 0]
    assert(len(sink_index)==1) #to make sure only one elecemt is negative 
    sink_index = sink_index[0]
    # 4. Check if the sum of b is close to zero
    b_sum = np.sum(b)
    if not np.isclose(b_sum, 0.0):
        print(f"Warning: Sum of vector b is not zero ({b_sum}). "
              "Adjusting b to ensure a valid solution exists for the Laplacian.")
        b -= b_sum / v

    # 5. Solve the linear system Lx = b using Conjugate Gradient
    print("5. Solving the linear system Lx = b...")
    
    # Start timer
    start_time = time.perf_counter()
    
    x, exit_code = cg(a, b, tol=1e-8)

    # End timer
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # shifting

    x = x - x[sink_index]
    
    print(f"Solver time: {elapsed_time:.4f} seconds ")

    # 6. Verify the solution
    if exit_code == 0:
        print("Solver converged successfully.")
        
        # Calculate the residual
        Lx = L.dot(x)
        residual_norm = np.linalg.norm(Lx - b)
        b_norm = np.linalg.norm(b)

        relative_residual = residual_norm / b_norm if b_norm != 0 else 0
        
        print(f"Relative residual ||Lx - b|| / ||b||: {relative_residual:.10f}")
        
        # 7. Save the solution to the output file
        print(f"\n7. Saving solution to '{output_filename}'...")
        try:
            with open(output_filename, 'w') as f:
                for val in x:
                    f.write(f"{val}\n")
            print("Solution saved successfully. ")
        except IOError:
            print(f"Error: Could not write to the file '{output_filename}'.")
            
    else:
        print(f"Solver failed to converge with exit code: {exit_code}")
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python solve_laplacian.py <input_filename> <output_filename>")
    else:
        input_filename = sys.argv[1]
        output_filename = sys.argv[2]
        solve_laplacian(input_filename, output_filename)