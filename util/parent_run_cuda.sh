#!/bin/bash
module load cuda/13.0
cd /home/cse/phd/csz208509/k-WalkLSolve-master/util
BASE_DIR="/home/cse/phd/csz208509/k-WalkLSolve-master/data"

# Array of all dataset keys
DATASETS=( "wikibug" "wikioc" "wikisr" "brain" "gene" "holl09" "holl11" "asSki") 


# Compile binary if not present
if [ ! -f "pcg_solver" ]; then
    echo "Compiling PCG solver..."
    nvcc -O3 Lx-b_cuda.cu -o pcg_solver
    if [ $? -ne 0 ]; then
        echo "Compilation failed!"
        exit 1
    fi
fi

# Iterate through each dataset sequentially
for DATASET_NAME in "${DATASETS[@]}"; do
    echo "=========================================="
    echo " Processing Dataset: $DATASET_NAME"
    echo "=========================================="

    case "$DATASET_NAME" in
        "wikibug")
            INPUT_GRAPH_PATH="${BASE_DIR}/datasets/wiki-bug/wiki_bug_csr_b5-sink" 
            OUTPUT_FILE_PATH="${BASE_DIR}/outputs/wiki-bug/wikibug_b5_pcg_sol"
            ;;
        "wikioc")
            INPUT_GRAPH_PATH="${BASE_DIR}/datasets/wiki_oc/wiki_oc_csr_b5-sink" 
            OUTPUT_FILE_PATH="${BASE_DIR}/outputs/wiki_oc/wikioc_b5_pcg_sol"
            ;;
        "wikisr")
            INPUT_GRAPH_PATH="${BASE_DIR}/datasets/wiki-sr/wiki_sr_csr_b5-sink" 
            OUTPUT_FILE_PATH="${BASE_DIR}/outputs/wiki-sr/wikisr_b5_pcg_sol" 
            ;;
        "brain")
            INPUT_GRAPH_PATH="${BASE_DIR}/datasets/brain/brain_csr_b5-sink" 
            OUTPUT_FILE_PATH="${BASE_DIR}/outputs/brain/brain_b5_pcg_sol" 
            ;;
        "gene")
            INPUT_GRAPH_PATH="${BASE_DIR}/datasets/gene/gene_csr_b5-sink" 
            OUTPUT_FILE_PATH="${BASE_DIR}/outputs/gene/gene_b5_pcg_sol" 
            ;;
        "holl09")
            INPUT_GRAPH_PATH="${BASE_DIR}/datasets/holl09/holl09_csr_b5-sink" 
            OUTPUT_FILE_PATH="${BASE_DIR}/outputs/holl09/holl09_b5_pcg_sol" 
            ;;
        "holl11")
            INPUT_GRAPH_PATH="${BASE_DIR}/datasets/holl11/holl11_csr_b5-sink" 
            OUTPUT_FILE_PATH="${BASE_DIR}/outputs/holl11/holl11_b5_pcg_sol"
            ;;
        "asSki")
            INPUT_GRAPH_PATH="${BASE_DIR}/datasets/as_ski/as_ski_csr_b5-sink" 
            OUTPUT_FILE_PATH="${BASE_DIR}/outputs/as_ski/asSki_b5_pcg_sol" 
            ;;
    esac

    # Execute solver for current dataset
    ./pcg_solver "$INPUT_GRAPH_PATH" "$OUTPUT_FILE_PATH"
    
    echo "Finished $DATASET_NAME"
    echo ""
done

echo "All datasets processed successfully."

