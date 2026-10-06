import numpy as np
from top_r import extract_top_r, evaluate_round_metrics

def main():
    # --- 1. Simulation Parameters ---
    V = 20  # Total number of nodes in the graph
    r = 5   # Number of top elements to extract
    round_num = 2

    print(f"--- Simulating Graph Data (V={V}, r={r}) ---")
    
    # --- 2. Mock Data: Previous Round (Round 1) ---
    # Simulate random visit counts between 0 and 100
    prev_cnt = np.random.randint(0, 100, size=V)
    prev_idx, prev_vals = extract_top_r(prev_cnt, r)
    
    # Simulate previous iteration's relevance/probability values (eta)
    prev_eta = np.random.rand(r) 

    # --- 3. Mock Data: Current Round (Round 2) ---
    # Simulate slightly different visit counts for the current round
    curr_cnt = np.random.randint(0, 100, size=V)
    curr_idx, curr_vals = extract_top_r(curr_cnt, r)
    
    # Simulate current iteration's relevance/probability values (eta)
    curr_eta = np.random.rand(r)

    # --- 4. Display the Generated Data ---
    print("\n[Previous Round (Round 1) Top R]")
    print(f"Indices: {prev_idx}")
    print(f"Counts:  {prev_vals}")
    print(f"Eta:     {np.round(prev_eta, 4)}")

    print("\n[Current Round (Round 2) Top R]")
    print(f"Indices: {curr_idx}")
    print(f"Counts:  {curr_vals}")
    print(f"Eta:     {np.round(curr_eta, 4)}")

    # --- 5. Evaluate Metrics ---
    # Pass the data to the evaluation function imported from top_r.py
    ndcg_score = evaluate_round_metrics(
        round_num=round_num,
        r=r,
        curr_idx=curr_idx,
        curr_eta=curr_eta,
        prev_idx=prev_idx,
        prev_eta=prev_eta
    )

if __name__ == "__main__":
    # Set a random seed for reproducible results during testing
    np.random.seed(42)
    main()
# How to run it:
# Ensure both top_r.py (containing the two functions from the previous step) and main.py are in the same directory.
# Run the script from your terminal:
# python main.py