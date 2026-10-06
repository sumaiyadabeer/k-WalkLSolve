#!/bin/bash
cd ..
cd include
ls
# Define the config file path
CONFIG_FILE="config.h"

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found!"
    exit 1
fi

# Function to update a #define value in the config file
update_define() {
    local key="$1"
    local value="$2"
    sed -i "s|#define $key .*|#define $key $value|" "$CONFIG_FILE"
}

read -p "Select GPU: " gpu_count
update_define "GPU_COUNT" "$gpu_count"

read -p "Enter DATASET_NAME (wikibug, wikioc, wikisr, brain, gene, holl09, holl11, asSki): " DATASET_NAME

BASE_DIR="/home/cse/phd/csz208509/k-WalkLSolve-master/data"

# Define updated input graph and output paths based on exact directory layout
if [[ "$DATASET_NAME" == "wikibug" ]]; then
    d=17
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/wiki-bug/wiki_bug_csr_b11-sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/wiki-bug/wikibug_b11_" 
elif [[ "$DATASET_NAME" == "wikioc" ]]; then
    d=9
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/wiki_oc/wiki_oc_csr_b11-sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/wiki_oc/wikioc_b11_" 
elif [[ "$DATASET_NAME" == "wikisr" ]]; then
    d=10
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/wiki-sr/wiki_sr_csr_b11-sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/wiki-sr/wikisr_b11_" 
elif [[ "$DATASET_NAME" == "brain" ]]; then
    d=6
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/brain/brain_csr_b11-sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/brain/brain_b11_" 
elif [[ "$DATASET_NAME" == "gene" ]]; then
    d=6
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/gene/gene_csr_b11-sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/gene/gene_b11_" 
elif [[ "$DATASET_NAME" == "holl09" ]]; then
    d=6
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/holl09/holl09_csr_b11-sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/holl09/holl09_b11_" 
elif [[ "$DATASET_NAME" == "holl11" ]]; then
    d=6
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/holl11/holl11_csr_b11-sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/holl11/holl11_b11_" 
elif [[ "$DATASET_NAME" == "asSki" ]]; then
    d=25
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/as_ski/as_ski_csr_b11-sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/as_ski/asSki_b11_" 
else
    echo "Unknown dataset: $DATASET_NAME"
    exit 1
fi

# 1. Hardcode SOLVER_COUNT to 1 (kwalk)
update_define "SOLVER_COUNT" "1"

# 2. Hardcode MAX_EPOCH as d^3
max_epoch=$(( d * d * d ))
update_define "MAX_EPOCH" "$max_epoch"

# 3. Hardcode RESIDUAL_RUN to 0 (disabled)
update_define "RESIDUAL_RUN" "0"

# Prompt for kwalk solver configuration
read -p "Enter new value for K-walk :: NO_OF_MSJ: " no_of_msj
update_define "NO_OF_MSJ" "$no_of_msj"

update_define "INPUTgRAPH" "\"$NEW_INPUT_GRAPH_PATH\""
update_define "OpFileSolK" "\"${NEW_OPFILESOLK_PATH}${no_of_msj}\""

echo "Configuration updated successfully in $CONFIG_FILE."
 
cd ..
make
./bin/main_app