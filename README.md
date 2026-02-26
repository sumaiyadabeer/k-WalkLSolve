# $k$-WalkLSolve

$k$-WalkLSolve is a high-performance, CUDA-accelerated project designed to calculate network centralities and solve graph-based linear systems.  

The project includes the core GPU solvers written in CUDA C++ and a suite of Python evaluation scripts to measure ranking quality using metrics like Precision, Recall, F1-Score, NDCG, Spearman's Rho, and Kendall's Tau.

## Project Structure

* **`include/` & `src/`**: Contains the core CUDA C++ implementation.
    * `main.cu`: The main entry point that initializes the graph, runs the selected solver, and calculates residuals.
    * `modules/kwalk.cu`: Implements the stochastic $k$-Walk solver using `curand` to simulate message packets over the network.
    * `modules/helper.cu`: Handles File I/O for Compressed Sparse Row (CSR) matrices and computes Laplacian errors.
* **`util/`**: Python scripts for performance evaluation and rank comparisons.
    * `rank_creation.py`: Converts raw centrality scores into integer ranks.
    * `full_rank.py`: Calculates Spearman's Rho and Kendall's Tau, and generates scatter plots comparing algorithm ranks to true ranks.
    * `partial_rank.py`: Computes Top-$r$ metrics (Precision@$r$, Recall@$r$, F1-Score@$r$, and NDCG@$r$).

## Prerequisites

To build and run the C++ code, your system must have:
* **NVIDIA GPU** with CUDA support.
* **CUDA Toolkit** (specifically `nvcc`, `cuBLAS`, and `cuSPARSE` libraries).
* **C++17** compatible compiler.

For the Python evaluation scripts, you will need:
* Python 3.x
* `numpy`, `scipy`, `matplotlib`, `seaborn`

*(Note: You can install the Python dependencies by creating a `requirements.txt` file and running `pip install -r requirements.txt`)*

## Configuration & Building

Before compiling, you can configure the behavior of the application by editing `include/config.h`. 

**Key Configuration Flags (`include/config.h`):**
* `INPUTgRAPH` / `INPUTpmAT`: Paths to your input datasets in CSR format.
* `NO_OF_MSJ`: Number of message packets for the random walk.
* `MAX_EPOCH`: Maximum number of epochs for the random walk.
* `OpFileSolK` / `OpFileSolM`: Output file paths for the solver results.

**Building the Project:**
The project uses a standard Makefile. Depending on your specific GPU architecture, you may need to adjust the `CUDA_ARCH` flag in the `Makefile` (currently set to Turing `sm_70`).

To compile the project, run:
```bash
make clean
make all 
```


## Running the Solvers

Once compiled, execute the main application:
```bash
./bin/main_app
```
The application will read the graphs specified in config.h, execute the selected solver(s) on the configured GPU, and save the raw centrality scores to your designated output files.

## Evaluation & Metrics
After generating the raw scores (OpFileSolM and OpFileSolK), use the Python scripts in the util/ folder to evaluate the ranking performance.

### Step 1: 
Create RanksConvert the raw output scores into ranked lists. You can optionally ignore source/sink nodes.
```bash 
python util/rank_creation.py --mFile path/to/mm_output --kFile path/to/kwalk_output --ignrSrc True
```

### Step 2: 
Full Rank CorrelationGenerate a scatter plot and calculate Spearman's Rho and Kendall's Tau between the MM (ground truth) and $k$-Walk rankings.
```bash
python util/full_rank.py --mRank path/to/mm_rank --kRank path/to/kwalk_rank --sctrPlt output_plot.png
```

### Step 3:
Partial Rank MetricsEvaluate the Top-$r$ accuracy (Precision, Recall, F1, NDCG). The script defaults to evaluating at $r = 25\%, 20\%, 15\%$.
```bash
python util/partial_rank.py --mRank path/to/mm_rank --kRank path/to/kwalk_rank
```

## Python Requirements (requirements.txt)
To ensure the evaluation scripts run smoothly, create a requirements.txt file in the root directory.Note on Python Versions: The rank_creation.py script uses distutils.util for parsing boolean arguments. If you are using Python 3.12 or newer, distutils has been removed from the standard library, so setuptools is included below to provide the necessary fallback.

```text
---Evaluation script dependencies---
numpy
scipy
matplotlib
seaborn
```
