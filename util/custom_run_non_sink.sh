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
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/non_sink/wikibug_non_sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/non_sink/wikibug" 
elif [[ "$DATASET_NAME" == "wikioc" ]]; then
    d=9
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/non_sink/wikioc_non_sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/non_sink/wikioc" 
elif [[ "$DATASET_NAME" == "wikisr" ]]; then
    d=10
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/non_sink/wikisr_non_sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/non_sink/wikisr" 
elif [[ "$DATASET_NAME" == "brain" ]]; then
    d=6
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/non_sink/brain_non_sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/non_sink/brain" 
elif [[ "$DATASET_NAME" == "gene" ]]; then
    d=6
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/non_sink/gene_non_sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/non_sink/gene" 
elif [[ "$DATASET_NAME" == "holl09" ]]; then
    d=6
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/non_sink/holl09_non_sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/non_sink/holl09" 
elif [[ "$DATASET_NAME" == "holl11" ]]; then
    d=6
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/non_sink/holl11_non_sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/non_sink/holl11" 
elif [[ "$DATASET_NAME" == "asSki" ]]; then
    d=25
    NEW_INPUT_GRAPH_PATH="${BASE_DIR}/datasets/non_sink/asski_non_sink" 
    NEW_OPFILESOLK_PATH="${BASE_DIR}/outputs/non_sink/asski" 
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
