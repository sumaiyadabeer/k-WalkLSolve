import numpy as np
from typing import List, Tuple
import os
import argparse

def calculate_precision_recall_f1_ndcg(
    R_a: np.ndarray,
    R_t: np.ndarray,
    r: int
) -> Tuple[float, float, float, float]:
    """
    Calculates Precision@r, Recall@r, F1-Score@r, and NDCG@r for a ranking algorithm.

    Args:
        R_a (np.ndarray): NumPy array of algorithm's assigned ranks for each item.
                          R_a[i] is the rank of item_i by the algorithm.
                          (e.g., [1, 3, 2] means item 0 is rank 1, item 1 is rank 3, item 2 is rank 2).
        R_t (np.ndarray): NumPy array of true ranks for each item (ground truth).
                          R_t[i] is the true rank of item_i.
                          (e.g., [1, 2, 3] means item 0 is true rank 1, item 1 is true rank 2, item 2 is true rank 3).
        r (int): The cutoff rank (e.g., @5, @10).

    Returns:
        Tuple[float, float, float, float]: (precision@r, recall@r, f1_score@r, ndcg@r)
    """
    if not (isinstance(R_a, np.ndarray) and isinstance(R_t, np.ndarray)):
        raise TypeError("Input R_a and R_t must be NumPy arrays.")
    if R_a.shape != R_t.shape:
        raise ValueError("Input rank vectors R_a and R_t must have the same shape.")
    if not np.issubdtype(R_a.dtype, np.integer) or not np.issubdtype(R_t.dtype, np.integer):
        print("Warning: Ranks should ideally be integers. Converting to int if needed.")
        R_a = R_a.astype(int)
        R_t = R_t.astype(int)
    if r <= 0:
        raise ValueError("r (cutoff) must be a positive integer.")

    num_items = len(R_a)
    
    # --- 1. Derive True Relevance Scores from R_t ---
    # We need to map item_id to its true relevance.
    # For P/R/F1: Binary relevance (1 if true rank <= r, else 0)
    # For NDCG: Graded relevance (higher true rank implies lower relevance score)

    # Dictionary to store true rank and derived relevance for each item_id
    item_info = {}
    for i in range(num_items):
        item_info[i] = {
            'true_rank': R_t[i],
            'algo_rank': R_a[i],
            'binary_relevance': 1 if R_t[i] <= r else 0, # Assuming items within true top-r are relevant
            'graded_relevance': (num_items - R_t[i] + 1) #  Higher true rank -> lower score. Max is 'N', Min is '1'.
        }
    
    # --- 2. Get Algorithm's Ranked List of Items ---
    # Create a list of (algo_rank, item_id) tuples and sort by algo_rank
    algo_ranked_items_with_ranks = sorted([(info['algo_rank'], item_id) 
                                           for item_id, info in item_info.items()], 
                                          key=lambda x: x[0])
    
    # Extract just the item_ids in the algorithm's ranked order
    algo_ranked_item_ids = [item_id for rank, item_id in algo_ranked_items_with_ranks]

    # --- 3. Calculate Precision@r, Recall@r, F1-score@r ---

    # Get the top 'r' items from the algorithm's ranking
    top_r_algo_items = algo_ranked_item_ids[:r]

    # Count relevant items in the algorithm's top 'r'
    relevant_in_top_r = 0
    for item_id in top_r_algo_items:
        if item_info[item_id]['binary_relevance'] == 1:
            relevant_in_top_r += 1

    # Total true relevant items (based on our binary relevance definition)
    total_true_relevant = sum(info['binary_relevance'] for info in item_info.values())
    #print("relevant_in_top_r: ",relevant_in_top_r,"total_true_relevant: ", total_true_relevant, "r: ", r )

    # Precision@r
    precision = relevant_in_top_r / r if r > 0 else 0.0

    # Recall@r
    recall = relevant_in_top_r / total_true_relevant if total_true_relevant > 0 else 0.0

    # F1-score@r
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # --- 4. Calculate NDCG@r ---

    # Calculate Discounted Cumulative Gain (DCG) for the algorithm's top 'r'
    dcg = 0.0
    for i in range(r):
        if i < len(algo_ranked_item_ids): # Ensure we don't go out of bounds if r > num_items
            item_id = algo_ranked_item_ids[i]
            relevance = item_info[item_id]['binary_relevance']
            dcg += (2**relevance - 1) / np.log2(i + 2) # i+1 for 1-based indexing in log (position 1 is log2(2))
        else:
            break # No more items to consider if r is too large

    # Calculate Ideal Discounted Cumulative Gain (IDCG) for the true top 'r'
    # Sort items by their true graded relevance in descending order to get ideal ranking
    ideal_ranked_items_with_relevance = sorted([(info['binary_relevance'], item_id) 
                                                for item_id, info in item_info.items()], 
                                               key=lambda x: x[0], reverse=True)
    
    idcg = 0.0
    for i in range(r):
        if i < len(ideal_ranked_items_with_relevance):
            relevance = ideal_ranked_items_with_relevance[i][0]
            idcg += (2**relevance - 1) / np.log2(i + 2)
        else:
            break

    # NDCG@r
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, f1_score, ndcg

def read_single_column_data_to_numpy_array(file_path, dtype=float):
    """
    Reads a text file with a single column into a NumPy array.

    Args:
        file_path (str): The path to the input text file.
        dtype (type): The desired data type for the array elements (e.g., float, int).
                      Defaults to float.

    Returns:
        np.ndarray: A NumPy array containing the data from the single column.
                    Returns None if the file cannot be read or contains no valid data.
    """
    column_values = []

    if not os.path.exists(file_path):
        print(f"ERROR: File not found at '{file_path}'")
        return None

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip() # Remove leading/trailing whitespace
                if not line: # Skip empty lines
                    continue

                parts = line.split() # Split by any whitespace

                if len(parts) != 1:
                    print(f"WARNING: Skipping line {line_num + 1} in '{file_path}' - expected 1 column, got {len(parts)}: '{line}'")
                    continue

                try:
                    val = dtype(parts[0])
                    column_values.append(val)
                except ValueError:
                    print(f"WARNING: Skipping line {line_num + 1} in '{file_path}' due to data conversion error to {dtype.__name__}: '{line}'")
                    continue

    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading '{file_path}': {e}")
        return None

    if not column_values:
        print(f"WARNING: No valid data found in '{file_path}'. File might be empty or all lines malformed.")
        return None

    data_array = np.array(column_values, dtype=dtype)
    return data_array


parser = argparse.ArgumentParser()
parser.add_argument('--mRank', dest='fileM', type=str, help='Add centrality from mm')
parser.add_argument('--kRank', dest='fileK', type=str, help='Add centrality from KS')
args = parser.parse_args()


# read actual eta in np.array
# read algo eta in np.array

filerankM = args.fileM #"/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug_new/wikibug_new_M_relay_rank"
filerankA = args.fileK #"/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug_new/wikibug_new_K_relay_rank"
rank_a = read_single_column_data_to_numpy_array(filerankM, dtype=int)
rank_k = read_single_column_data_to_numpy_array(filerankA, dtype=int)

n=rank_a.size

# reverse_rank_a =[]
# reverse_rank_k =[]
# for i in range(n):
#     reverse_rank_a.append(n-rank_a[i])
#     reverse_rank_k.append(n-rank_k[i])
# rank_a = np.array(reverse_rank_a)
# rank_k = np.array(reverse_rank_k)


# save ranks for future

r_list = [25, 20, 15] #[25, 20, 15, 10, 5, 1, 0.1]
for p in r_list:
    r = int(n*(p/100))+1
    prec, rec, f1, ndcg = calculate_precision_recall_f1_ndcg(rank_k, rank_a, r)
    # print(f"  R_t (true ranks): {rank_a}")
    # print(f"  R_k (algo ranks): {rank_k}")
    print(prec, ",", rec, ",", f1, ",", ndcg, end=",")
    # print(f"  Precision@{r}: {prec:.4f}") # Expected: 1.0 (all top 5 are truly top 5)
    # print(f"  Recall@{r}:    {rec:.4f}")    # Expected: 1.0 (all 5 true top relevant items are found)
    # print(f"  F1-Score@{r}:  {f1:.4f}")      # Expected: 1.0
    # print(f"  NDCG@{r}:      {ndcg}")     # Expected: 1.0 (perfect ranking)

print("")
# print("\n \n Precision@r \t Recall@r \t F1-Score@r \t NDCG@r")
for p in r_list:
    r = int(n*(p/100))+1
    # print("r is: ", p, " and num of items: ", r)

# # exit()

# # --- Example Usage ---

# # Example data: n=10 items
# # R_t[i] = true rank of item_i
# # R_a[i] = algorithm's rank of item_i
# # Note: Lower rank number means higher in the ranking (rank 1 is best)

# # Example 1: Perfect Ranking
# R_t_perfect = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Item 0 is rank 1, Item 1 is rank 2, etc.
# R_a_perfect = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Algorithm ranks items perfectly

# r_val_1 = 5
# print(f"--- Example 1: Perfect Ranking (n={len(R_t_perfect)}, r={r_val_1}) ---")
# prec, rec, f1, ndcg = calculate_precision_recall_f1_ndcg(R_a_perfect, R_t_perfect, r_val_1)
# print(f"  R_t (true ranks): {R_t_perfect}")
# print(f"  R_a (algo ranks): {R_a_perfect}")
# print(f"  Precision@{r_val_1}: {prec:.4f}") # Expected: 1.0 (all top 5 are truly top 5)
# print(f"  Recall@{r_val_1}:    {rec:.4f}")    # Expected: 1.0 (all 5 true top relevant items are found)
# print(f"  F1-Score@{r_val_1}:  {f1:.4f}")      # Expected: 1.0
# print(f"  NDCG@{r_val_1}:      {ndcg:.4f}")     # Expected: 1.0 (perfect ranking)

# # Example 2: Imperfect Ranking with some swaps
# R_t_imperfect = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# R_a_imperfect = np.array([3, 1, 2, 6, 5, 4, 8, 7, 10, 9]) # Item 0: rank 3, Item 1: rank 1, etc.

# r_val_2 = 5
# print(f"\n--- Example 2: Imperfect Ranking (n={len(R_t_imperfect)}, r={r_val_2}) ---")
# prec, rec, f1, ndcg = calculate_precision_recall_f1_ndcg(R_a_imperfect, R_t_imperfect, r_val_2)
# print(f"  R_t (true ranks): {R_t_imperfect}")
# print(f"  R_a (algo ranks): {R_a_imperfect}")
# print(f"  Precision@{r_val_2}: {prec:.4f}")
# print(f"  Recall@{r_val_2}:    {rec:.4f}")
# print(f"  F1-Score@{r_val_2}:  {f1:.4f}")
# print(f"  NDCG@{r_val_2}:      {ndcg:.4f}")

# # Manual check for Example 2, r=5:
# # True top 5 items (by R_t): Item 0 (R_t=1), Item 1 (R_t=2), Item 2 (R_t=3), Item 3 (R_t=4), Item 4 (R_t=5)
# # Algo top 5 items (by R_a):
# #   Item 1 (R_a=1, R_t=2) - Relevant
# #   Item 2 (R_a=2, R_t=3) - Relevant
# #   Item 0 (R_a=3, R_t=1) - Relevant
# #   Item 4 (R_a=4, R_t=5) - Relevant
# #   Item 3 (R_a=5, R_t=4) - Relevant
# # All 5 items in algo top 5 are true top 5 relevant items. So P@5=5/5=1.0, R@5=5/5=1.0, F1=1.0.
# # NDCG will be less than 1.0 because the order within the top 5 is not ideal (e.g., Item 0, true rank 1, is ranked 3rd by algo).

# # Example 3: Different r and more items
# R_t_large = np.arange(1, 21)
# R_a_large = np.copy(R_t_large)
# np.random.shuffle(R_a_large) # Randomize for a realistic test case

# r_val_3 = 10
# print(f"\n--- Example 3: Larger Dataset, Random Ranks (n={len(R_t_large)}, r={r_val_3}) ---")
# prec, rec, f1, ndcg = calculate_precision_recall_f1_ndcg(R_a_large, R_t_large, r_val_3)
# print(f"  Precision@{r_val_3}: {prec:.4f}")
# print(f"  Recall@{r_val_3}:    {rec:.4f}")
# print(f"  F1-Score@{r_val_3}:  {f1:.4f}")
# print(f"  NDCG@{r_val_3}:      {ndcg:.4f}")

# # Example 4: Edge case - r > num_items
# r_val_4 = 15
# print(f"\n--- Example 4: r > num_items (n={len(R_t_perfect)}, r={r_val_4}) ---")
# prec, rec, f1, ndcg = calculate_precision_recall_f1_ndcg(R_a_perfect, R_t_perfect, r_val_4)
# print(f"  Precision@{r_val_4}: {prec:.4f}")
# print(f"  Recall@{r_val_4}:    {rec:.4f}")
# print(f"  F1-Score@{r_val_4}:  {f1:.4f}")
# print(f"  NDCG@{r_val_4}:      {ndcg:.4f}")
