import numpy as np

def extract_top_r(cnt: np.ndarray, r: int):
    """
    Extracts the indices and values of the top 'r' elements.
    Uses introselect (argpartition) for O(V) time complexity instead of O(V * r).
    """
    V = len(cnt)
    r_actual = min(r, V)
    
    # Partition array so the top r_actual elements are at the end, O(V) time
    idx = np.argpartition(cnt, -r_actual)[-r_actual:]
    
    # argpartition doesn't guarantee sorted order, so we sort just those top r elements
    top_indices = idx[np.argsort(-cnt[idx])]
    top_values = cnt[top_indices]
    
    # Replicate the C++ behavior: pad with -1 if r > V
    if r > V:
        pad_width = r - V
        top_indices = np.pad(top_indices, (0, pad_width), constant_values=-1)
        top_values = np.pad(top_values, (0, pad_width), constant_values=-1)
        
    return top_indices, top_values

def evaluate_round_metrics(round_num: int, r: int, 
                           curr_idx: np.ndarray, curr_eta: np.ndarray, 
                           prev_idx: np.ndarray, prev_eta: np.ndarray) -> float:
    print("\n==================================================")
    print(f" METRICS COMPARISON FOR ROUND {round_num} (Top R = {r})")
    print("==================================================")

    # 1. Build a quick lookup map (dictionary is extremely fast for this in Python)
    prev_relevance_map = dict(zip(prev_idx, prev_eta))

    # 2. Vectorized Graded NDCG Calculation
    # Look up relevances. Default to 0.0 if not found.
    relevances = np.array([prev_relevance_map.get(idx, 0.0) for idx in curr_idx[:r]])
    
    # Discounts = log2(i + 2). Note: np.arange is 0-indexed, so we add 2.
    discounts = np.log2(np.arange(2, r + 2))
    dcg = np.sum(relevances / discounts)

    # 3. Calculate Ideal DCG
    ideal_relevance = np.sort(prev_eta)[::-1][:r]
    idcg = np.sum(ideal_relevance / discounts[:len(ideal_relevance)])

    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

    # Print Results cleanly
    print(f" -> Graded NDCG:     {ndcg:.6f}")
    print("==================================================")

    return ndcg
