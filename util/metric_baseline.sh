#!/bin/bash

# --- Configuration & Paths ---
BASE_PATH="/home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs"
FOLDERS=("brain" ) #"as_ski" "wiki-bug" "gene" "holl09" "holl11" "wiki_oc" "wiki-sr"

# Variable for target baseline output filename to compare against pcg_sol
# Change this variable when evaluating different baseline output files (e.g., "kyng_sach_out", "pyamg_SA_out")
TARGET_SOLVER_FILE="pyaSA_SA"

RANK="_rank"
RELAY_RANK="_relay_rank"

echo "Starting rank evaluation loop..."

for folder in "${FOLDERS[@]}"; do
    folder_path="${BASE_PATH}/${folder}"
    
    # Reference file (pcg_sol) and target solver file
    centrality_m="${folder_path}/pcg_sol"
    centrality_k="${folder_path}/${TARGET_SOLVER_FILE}"
    
    # NOTE: If your target file has a folder-prefixed pattern (e.g., holl11_k131072, gene_k131072),
    # uncomment the line below:
    # centrality_k="${folder_path}/${folder}_k131072"

    scatter_plt_file="${centrality_k}_scatter.png"
    scatter_plt_relay_file="${centrality_k}_scatter_relay.png"

    echo "=================================================="
    echo "Processing Dataset: ${folder}"
    echo "Reference (m): ${centrality_m}"
    echo "Target    (k): ${centrality_k}"
    echo "=================================================="

    # Check if both output files exist before running rank evaluation
    if [[ ! -f "$centrality_m" ]]; then
        echo "Error: Reference file '${centrality_m}' not found. Skipping ${folder}..."
        continue
    fi

    if [[ ! -f "$centrality_k" ]]; then
        echo "Error: Target file '${centrality_k}' not found. Skipping ${folder}..."
        continue
    fi

    echo "source vertices considered"
    python rank_creation.py --mFile "$centrality_m" --kFile "$centrality_k" --ignrSrc "False"
    python full_rank.py --mRank "${centrality_m}${RANK}" --kRank "${centrality_k}${RANK}" --sctrPlt "$scatter_plt_file" 
    python partial_rank.py --mRank "${centrality_m}${RANK}" --kRank "${centrality_k}${RANK}"

done

echo "bash script is complete"
