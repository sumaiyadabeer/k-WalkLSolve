import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import argparse
import distutils.util


def read_two_column_data_to_numpy_arrays(file_path):
    """
    Reads a text file with two space-separated columns into two separate NumPy arrays.

    Args:
        file_path (str): The path to the input text file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                      (first_column_data, second_column_data).
                                      Returns (None, None) if the file cannot be read
                                      or contains no valid data.
    """
    first_column_values = []
    second_column_values = []

    if not os.path.exists(file_path):
        print(f"ERROR: File not found at '{file_path}'")
        return None, None

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip() # Remove leading/trailing whitespace
                if not line: # Skip empty lines
                    continue

                parts = line.split() # Split by any whitespace

                if len(parts) != 2:
                    print(f"WARNING: Skipping line {line_num + 1} in '{file_path}' - expected 2 columns, got {len(parts)}: '{line}'")
                    continue

                try:
                    first_val = float(parts[0])
                    second_val = int(parts[1])

                    first_column_values.append(first_val)
                    second_column_values.append(second_val)
                except ValueError:
                    print(f"WARNING: Skipping line {line_num + 1} in '{file_path}' due to data conversion error: '{line}'")
                    continue

    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading '{file_path}': {e}")
        return None, None

    if not first_column_values:
        print(f"WARNING: No valid data found in '{file_path}'. File might be empty or all lines malformed.")
        return None, None

    # Convert lists to NumPy arrays
    first_array = np.array(first_column_values, dtype=float)
    second_array = np.array(second_column_values, dtype=int)

    return first_array, second_array

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

def get_ranks_from_vector(data_vector, method='average', order='descending'):
    """
    Calculates the rank of each element in a given vector.

    Args:
        data_vector (list or np.array): A 1D vector of float values.
        method (str): Specifies how to handle ties. Options include:
                      'average' (default): Assigns the average of the ranks that
                                           would have been assigned to all the tied values.
                      'min': Assigns the minimum rank to all tied values.
                      'max': Assigns the maximum rank to all tied values.
                      'dense': Assigns ranks like 'min' but increments ranks
                               by 1 for each unique value (no gaps).
                      'ordinal': Assigns unique ranks based on original order for ties.
        order (str): Specifies the ranking order.
                     'ascending' (default): Smallest value gets rank 1.
                     'descending': Largest value gets rank 1.

    Returns:
        np.array: A NumPy array containing the rank of each element,
                  maintaining the original order of the input vector.
    """
    if not isinstance(data_vector, (list, np.ndarray)):
        raise TypeError("Input must be a list or a NumPy array.")

    data_vector = np.asarray(data_vector, dtype=float)

    if order == 'ascending':
        ranks = rankdata(data_vector, method=method)
    elif order == 'descending':
        # For descending, rank the negative values in ascending order
        ranks = rankdata(-data_vector, method=method)
    else:
        raise ValueError("Order must be 'ascending' or 'descending'.")

    return ranks


# ----------- correlation coefficients  ----------- #


parser = argparse.ArgumentParser()
parser.add_argument('--mFile', dest='fileM', type=str, help='Add centrality from mm')
parser.add_argument('--kFile', dest='fileK', type=str, help='Add centrality from KS')
parser.add_argument('--ignrSrc', dest='ignoreSource', type=distutils.util.strtobool, help='Add whether source are ignored')
args = parser.parse_args()


# read actual eta in np.array
# read algo eta in np.array
filepathM = args.fileM #"/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug_new/wikibug_new_m"
filepathA = args.fileK #"/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug_new/wikibug_new_k"

# logic to remove source sink
ignore_source_sink  = args.ignoreSource


#eta_a, source_sink = read_two_column_data_to_numpy_arrays(filepathM)

eta_a = read_single_column_data_to_numpy_array(filepathM, dtype=float)
eta_k = read_single_column_data_to_numpy_array(filepathA, dtype=float)
filerankM = filepathM + "_rank"
filerankA = filepathA + "_rank"

#removed ignored source sink logic   
    
# get actual rank in np.array (MM)
rank_a = get_ranks_from_vector(eta_a, method='min', order='descending')
# get algo rank in np.array (KS)
rank_k = get_ranks_from_vector(eta_k, method='min', order='descending')
# save ranks for future
#print(rank_k, source_sink, rank_a)
np.savetxt(filerankA, rank_k.astype(int), fmt='%i', delimiter='\n')
np.savetxt(filerankM, rank_a.astype(int), fmt='%i', delimiter='\n')
