main.jl
# Include the file containing your functions
include("top_r.jl")
using Random

function main()
    # --- 1. Simulation Parameters ---
    V = 20  # Total number of nodes in the graph
    r = 5   # Number of top elements to extract
    round_num = 2

    println("--- Simulating Graph Data (V=$V, r=$r) ---")
    
    # --- 2. Mock Data: Previous Round (Round 1) ---
    # Simulate random visit counts between 0 and 100
    prev_cnt = rand(0:100, V)
    prev_idx, prev_vals = extract_top_r(prev_cnt, r)
    
    # Simulate previous iteration's relevance/probability values (eta)
    # Using Float32 to match the types expected in the evaluate function
    prev_eta = rand(Float32, r) 

    # --- 3. Mock Data: Current Round (Round 2) ---
    # Simulate slightly different visit counts for the current round
    curr_cnt = rand(0:100, V)
    curr_idx, curr_vals = extract_top_r(curr_cnt, r)
    
    # Simulate current iteration's relevance/probability values (eta)
    curr_eta = rand(Float32, r)

    # --- 4. Display the Generated Data ---
    println("\n[Previous Round (Round 1) Top R]")
    println("Indices: ", prev_idx)
    println("Counts:  ", prev_vals)
    println("Eta:     ", round.(prev_eta, digits=4))

    println("\n[Current Round (Round 2) Top R]")
    println("Indices: ", curr_idx)
    println("Counts:  ", curr_vals)
    println("Eta:     ", round.(curr_eta, digits=4))

    # --- 5. Evaluate Metrics ---
    # Pass the data to the evaluation function included from top_r.jl
    ndcg_score = evaluate_round_metrics(
        round_num,
        r,
        curr_idx,
        curr_eta,
        prev_idx,
        prev_eta
    )
end

# Set a random seed for reproducible results during testing
Random.seed!(42)

# Execute the main function
main()
# How to run it:
# Ensure both top_r.jl (containing your extract_top_r and evaluate_round_metrics functions) and main.jl are in the same directory.
# Run the script from your terminal:
# julia main.jl