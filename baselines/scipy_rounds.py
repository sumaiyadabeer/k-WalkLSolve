import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.linalg import cg

def compute_consecutive_ndcg(prev_Dx, curr_Dx, r=50):
    r = min(r, len(prev_Dx))
    
    # Ideal ranking order based on previous round's Dx
    ideal_perm = np.argsort(prev_Dx)[::-1][:r]
    discounts = 1.0 / np.log2(np.arange(2, r + 2))
    idcg = np.sum(prev_Dx[ideal_perm] * discounts)
    
    if idcg == 0.0:
        return 1.0
        
    # Current round's top-r ranking evaluated against previous round's scores
    curr_perm = np.argsort(curr_Dx)[::-1][:r]
    dcg = np.sum(prev_Dx[curr_perm] * discounts)
    
    return dcg / idcg

def solve_laplacian_rounds(input_filename, output_filename, num_iter):
    print(f"1. Reading data from '{input_filename}'...")
    try:
        with open(input_filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
        return

    # Parse input graph
    try:
        v, e = map(int, lines[0].strip().split())
        row_ptr = list(map(int, lines[1].strip().split()))
        col_offset = list(map(int, lines[2].strip().split()))
        values = list(map(float, lines[3].strip().split()))
        b = np.array(list(map(float, lines[4].strip().split())))
    except (ValueError, IndexError):
        print("Error: The input file format is incorrect.")
        return

    print(f"Parsed graph with {v} vertices and {e} edges.")

    # Construct sparse Laplacian matrix and extract degrees D
    A = csr_matrix((values, col_offset, row_ptr), shape=(v, v))
    L = csr_matrix(csgraph.laplacian(A))
    degrees = L.diagonal()

    # Locate sink vertex
    sink_indices = np.where(b < 0)[0]
    assert len(sink_indices) == 1, "Expected exactly one sink vertex (negative entry in b)."
    sink_index = sink_indices[0]

    # Adjust b if sum is not zero
    b_sum = np.sum(b)
    if not np.isclose(b_sum, 0.0):
        b -= b_sum / v

    print("2. Starting iterative Conjugate Gradient solver across rounds...")
    
    x = np.zeros(v)
    prev_Dx = None
    max_rounds = 100
    ndcg_threshold = 0.95
    
    start_time = time.perf_counter()

    for round_num in range(1, max_rounds + 1):
        # Execute fixed iterations for the current round starting from previous solution state
        x_sol, exit_code = cg(L, b, x0=x, maxiter=num_iter, rtol=1e-12)
        x = x_sol  # Update working solution
        
        # Ground sink and calculate D * x
        x_shifted = x - x[sink_index]
        curr_Dx = degrees * x_shifted
        
        # Compute residual norm
        Lx = L.dot(x)
        res_norm = np.linalg.norm(Lx - b)
        
        if round_num == 1:
            print(f"Round 1 completed | Total Iterations: {num_iter} | Residual: {res_norm:.6e}")
            prev_Dx = curr_Dx.copy()
        else:
            ndcg_val = compute_consecutive_ndcg(prev_Dx, curr_Dx, r=50)
            print(f"Round {round_num} completed | Total Iterations: {round_num * num_iter} | Residual: {res_norm:.6e} | Consecutive NDCG@50: {ndcg_val:.6f}")
            
            if ndcg_val > ndcg_threshold:
                print(f"\nConvergence reached: Consecutive NDCG@50 > {ndcg_threshold} at round {round_num}.")
                prev_Dx = curr_Dx.copy()
                break
            
            prev_Dx = curr_Dx.copy()

    elapsed_time = time.perf_counter() - start_time
    print(f"Solver finished in {elapsed_time:.4f} seconds.")

    # Export degree-multiplied solution D * x
    print(f"\n3. Saving grounded D*x solution to '{output_filename}'...")
    try:
        with open(output_filename, 'w') as f:
            for val in prev_Dx:
                f.write(f"{val}\n")
        print("Solution saved successfully.")
    except IOError:
        print(f"Error: Could not write to output file '{output_filename}'.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python solve_laplacian_rounds.py <input_filename> <output_filename> <num_fixed_iterations>")
    else:
        input_filename = sys.argv[1]
        output_filename = sys.argv[2]
        num_iter = int(sys.argv[3])
        solve_laplacian_rounds(input_filename, output_filename, num_iter)
