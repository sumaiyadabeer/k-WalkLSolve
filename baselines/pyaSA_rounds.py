import sys
import time
import argparse
import numpy as np
from scipy.sparse import csr_matrix, csgraph
import pyamg

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

def read_graph(filename):
    with open(filename, 'r') as file1:
        Lines = file1.readlines()

    node_edge = [int(i) for i in Lines[0].split()]
    row_ptr = [int(i) for i in Lines[1].split()]
    col_off = [int(i) for i in Lines[2].split()]
    values = [float(i) for i in Lines[3].split()]
    b = [float(i) for i in Lines[4].split()]
    
    return node_edge, row_ptr, col_off, values, b

def main():
    parser = argparse.ArgumentParser(description="PyAMG Iterative Round Solver")
    parser.add_argument("-csr", "--csr", required=True, help="Path to input CSR graph file")
    parser.add_argument("-pyamFile", "--pyamFile", required=True, help="Path to output solution file")
    parser.add_argument("-num_iter", "--num_iter", type=int, default=6, help="Number of fixed AMG cycles per round")
    args = parser.parse_args()

    print(f"1. Reading input graph from '{args.csr}'...")
    node_edge, row_ptr, col_off, values, b = read_graph(args.csr)
    v = len(row_ptr) - 1
    
    # Construct sparse adjacency matrix and Laplacian
    L_adj = csr_matrix((values, col_off, row_ptr), shape=(v, v))
    a = csr_matrix(csgraph.laplacian(L_adj))
    degrees = a.diagonal()

    b = np.array(b, dtype=np.float64)
    sink_indices = np.where(b < 0)[0]
    assert len(sink_indices) == 1, "Expected exactly one sink vertex (negative entry in b)."
    sink_index = sink_indices[0]

    # Adjust b if sum is not zero
    b_sum = np.sum(b)
    if not np.isclose(b_sum, 0.0):
        b -= b_sum / v

    print("2. Building PyAMG Smoothed Aggregation hierarchy...")
    ml = pyamg.smoothed_aggregation_solver(a)

    print(f"3. Starting iterative AMG solver ({args.num_iter} cycles per round)...")
    
    x = np.zeros(v, dtype=np.float64)
    prev_Dx = None
    max_rounds = 100
    ndcg_threshold = 0.95
    
    start_time = time.perf_counter()

    for round_num in range(1, max_rounds + 1):
        # Perform fixed AMG cycles starting from previous solution x0=x
        res_list = []
        x = ml.solve(b, x0=x, maxiter=args.num_iter, tol=1e-12, residuals=res_list)
        
        # Ground sink and calculate D * x
        x_shifted = x - x[sink_index]
        curr_Dx = degrees * x_shifted
        
        # Calculate residual norm
        residual = b - a.dot(x)
        res_norm = np.linalg.norm(residual)
        
        if round_num == 1:
            print(f"Round 1 completed | Total Cycles: {args.num_iter} | Residual: {res_norm:.6e}")
            prev_Dx = curr_Dx.copy()
        else:
            ndcg_val = compute_consecutive_ndcg(prev_Dx, curr_Dx, r=50)
            print(f"Round {round_num} completed | Total Cycles: {round_num * args.num_iter} | Residual: {res_norm:.6e} | Consecutive NDCG@50: {ndcg_val:.6f}")
            
            if ndcg_val > ndcg_threshold:
                print(f"\nConvergence reached: Consecutive NDCG@50 > {ndcg_threshold} at round {round_num}.")
                prev_Dx = curr_Dx.copy()
                break
            
            prev_Dx = curr_Dx.copy()

    elapsed_time = time.perf_counter() - start_time
    print(f"Solver finished in {elapsed_time:.4f} seconds.")

    # Export grounded D * x solution vector
    output_path = args.pyamFile + "_SA"
    print(f"\n4. Saving grounded D*x solution to '{output_path}'...")
    try:
        with open(output_path, 'w') as f:
            for val in prev_Dx:
                f.write(f"{val}\n")
        print("Solution saved successfully.")
    except IOError:
        print(f"Error: Could not write to output file '{output_path}'.")

if __name__ == "__main__":
    main()
