import numpy as np
# from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import argparse

plt.rcParams.update({'font.size': 17})

# ----------- correlation coefficients  ----------- #

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


def calculate_spearmans_rho(R_a: np.ndarray, R_t: np.ndarray) -> tuple[float, float]:
    """
    Calculates Spearman's Rank Correlation Coefficient between two ranking vectors.

    Args:
        R_a (np.ndarray): NumPy array representing the ranks from an algorithm.
        R_t (np.ndarray): NumPy array representing the ground truth ranks.

    Returns:
        tuple[float, float]: A tuple containing:
            - The Spearman correlation coefficient (rho).
            - The p-value.
        Returns (np.nan, np.nan) if input vectors are empty or have different lengths.
    """
    if len(R_a) == 0 or len(R_t) == 0 or len(R_a) != len(R_t):
        print("Warning: Input rank vectors must be non-empty and of the same length.")
        return np.nan, np.nan

    rho, p_value = stats.spearmanr(R_a, R_t)
    return rho, p_value

def calculate_kendalls_tau(R_a: np.ndarray, R_t: np.ndarray) -> tuple[float, float]:
    """
    Calculates Kendall's Tau Correlation Coefficient between two ranking vectors.

    Args:
        R_a (np.ndarray): NumPy array representing the ranks from an algorithm.
        R_t (np.ndarray): NumPy array representing the ground truth ranks.

    Returns:
        tuple[float, float]: A tuple containing:
            - The Kendall's Tau correlation coefficient (tau).
            - The p-value.
        Returns (np.nan, np.nan) if input vectors are empty or have different lengths.
    """
    if len(R_a) == 0 or len(R_t) == 0 or len(R_a) != len(R_t):
        print("Warning: Input rank vectors must be non-empty and of the same length.")
        return np.nan, np.nan

    tau, p_value = stats.kendalltau(R_a, R_t)
    return tau, p_value

# --- Example Usage --- #

parser = argparse.ArgumentParser()
parser.add_argument('--mRank', dest='fileM', type=str, help='Add centrality from mm')
parser.add_argument('--kRank', dest='fileK', type=str, help='Add centrality from KS')
parser.add_argument('--sctrPlt', dest='ScatterPlot', type=str, help='Add name of scatterplot file')
# parser.add_argument('--diffPlt', dest='DifferencePlot', type=str, help='Add name of scatterplot file')
args = parser.parse_args()


# get actual rank in np.array
# get algo rank in np.array

filerankM = args.fileM #"/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug_new/wikibug_new_M_relay_rank"
filerankA = args.fileK #"/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug_new/wikibug_new_K_relay_rank"
rank_a = read_single_column_data_to_numpy_array(filerankM, dtype=int)
rank_k = read_single_column_data_to_numpy_array(filerankA, dtype=int)
# save ranks for future
n=rank_a.size
print(n,"  ", rank_k.size )
# ----------- Selected Graphical Methods to Assess Ranking Quality ---

# --- Plot 1: Scatter Plot of Algorithm Rank vs. True Rank (No joining lines, with y=x line) ---
print("\nGenerating Scatter Plot of Algorithm Rank vs. True Rank...")
plt.figure(figsize=(8, 8))

# Plot the scatter points: True Rank on x-axis, Algorithm Rank on y-axis
sns.scatterplot(x=rank_a, y=rank_k, alpha=0.6, s=50, color='purple')

# Add a dashed diagonal line representing perfect agreement (y = x)
plt.plot([1, n], [1, n], color='red', linestyle='--', label='Perfect Agreement ($R^* = R$)')

plt.xlabel('True Rank ($R^*$)', fontsize=30)
plt.xticks(fontsize=34 , rotation=90) #, 
plt.yticks(fontsize=34) 
plt.ylabel('k-WalkLSolve Rank ($R$)', fontsize=30)
# plt.title('Scatter Plot: $k$-WalkLSolve Rank vs. True Rank')
plt.grid(True, linestyle=':', alpha=0.7)
# plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(args.ScatterPlot)


# # --- Plot 2 (Previous Plot 3): Rank Difference per True Rank (Line Plot) ---
# print("Generating Rank Difference per True Rank Plot...")

# # Calculate the difference between algorithm rank and true rank for each item
# rank_differences = rank_k - rank_a

# plt.figure(figsize=(12, 6))
# # Plotting the rank differences against the true ranks (which are already sorted 1 to n)
# plt.plot(rank_a, rank_differences, marker='.', linestyle='--', alpha=0.7, color='green')
# plt.axhline(0, color='red', linestyle='--', label='Zero Difference')

# plt.xlabel('True Rank ($R_t$)')
# plt.ylabel('Rank Difference ($R_a - R_t$)')
# plt.title('Rank Difference for Each Item (Ordered by True Rank)')
# plt.legend()
# plt.grid(True, linestyle=':', alpha=0.7)
# plt.tight_layout()
# plt.savefig(args.DifferencePlot)

print("\nBoth selected plots generated successfully!")



print("caculation Spearman's ")
rho_partial, p_rho_partial = calculate_spearmans_rho(rank_k, rank_a)
print("calculating kendell's")
tau_partial, p_tau_partial = calculate_kendalls_tau(rank_k, rank_a)


print(f"Spearman's Rho: {rho_partial:.4f} (p-value: {p_rho_partial:.6f})")
print(f"Kendall's Tau:  {tau_partial:.4f} (p-value: {p_tau_partial:.6f})")
print("\n", rho_partial, ",", tau_partial, ",-", ",-", end=",")
# Expected: Positive values, less than 1.0
