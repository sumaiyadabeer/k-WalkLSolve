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
    # The 's' command in sed replaces the first occurrence of the pattern
    # The pattern matches '#define KEY' followed by any characters, up to the end of the line
    # It replaces the entire line with the new #define with the new value
    sed -i "s|#define $key .*|#define $key $value|" "$CONFIG_FILE"
}

read -p "Select GPU: " gpu_count
update_define "GPU_COUNT" "$gpu_count"

read -p "Enter DATASET_NAME (wikibug, wikioc, wikisr, brain, gene, holl09, holl11, asSki): " DATASET_NAME
# Define the new base path based on the dataset name
if [[ "$DATASET_NAME" == "wikibug" ]]; then
    NEW_INPUT_GRAPH_PATH="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikibug/bug_csr" 
    NEW_INPUT_PMAT_PATH="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikibug/bug_csr_P"  
    NEW_OPFILESOLK_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug/wikibug_k" 
    NEW_OPFILESOLM_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/wikibug/wikibug_m" 
elif [[ "$DATASET_NAME" == "wikioc" ]]; then
    NEW_INPUT_GRAPH_PATH="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikioc/oc_csr" 
    NEW_INPUT_PMAT_PATH="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikioc/oc_csr_P"  
    NEW_OPFILESOLK_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/wikioc/wikioc_k" 
    NEW_OPFILESOLM_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/wikioc/wikioc_m" 
elif [[ "$DATASET_NAME" == "wikisr" ]]; then
    NEW_INPUT_GRAPH_PATH="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikisr/out.wikipedia_link_sr_csr" 
    NEW_INPUT_PMAT_PATH="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikisr/out.wikipedia_link_sr_csr_P"  
    NEW_OPFILESOLK_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/wikisr/wikisr_k" 
    NEW_OPFILESOLM_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/wikisr/wikisr_m" 
elif [[ "$DATASET_NAME" == "brain" ]]; then
    NEW_INPUT_GRAPH_PATH="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/brain/brain.edges_csr" 
    NEW_INPUT_PMAT_PATH="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/brain/brain.edges_csr_P"  
    NEW_OPFILESOLK_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/brain/brain_k" 
    NEW_OPFILESOLM_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/brain/brain_m" 
elif [[ "$DATASET_NAME" == "gene" ]]; then
    NEW_INPUT_GRAPH_PATH="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/gene/gene.edges_csr" 
    NEW_INPUT_PMAT_PATH="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/gene/gene.edges_csr_P"  
    NEW_OPFILESOLK_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/gene/gene_k" 
    NEW_OPFILESOLM_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/gene/gene_m" 
elif [[ "$DATASET_NAME" == "holl09" ]]; then
    NEW_INPUT_GRAPH_PATH="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-09/holl09_csr" 
    NEW_INPUT_PMAT_PATH="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-09/holl09_csr_P"  
    NEW_OPFILESOLK_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/holl09/holl09_k" 
    NEW_OPFILESOLM_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/holl09/holl09_m" 
elif [[ "$DATASET_NAME" == "holl11" ]]; then
    NEW_INPUT_GRAPH_PATH="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-11/holl11_csr" 
    NEW_INPUT_PMAT_PATH="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-11/holl11_csr_P"  
    NEW_OPFILESOLK_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/holl11/holl11_k" 
    NEW_OPFILESOLM_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/holl11/holl11_m" 
elif [[ "$DATASET_NAME" == "asSki" ]]; then
    NEW_INPUT_GRAPH_PATH="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/as-ski/as-skitter.txt_csr" 
    NEW_INPUT_PMAT_PATH="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/as-ski/as-skitter.txt_csr_P"  
    NEW_OPFILESOLK_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/asSki/asSki_k" 
    NEW_OPFILESOLM_PATH="/home/sumaiya/FixedPS/kSolve/data/outputs/asSki/asSki_m" 
else
    echo "Unknown dataset: $DATASET_NAME"
    exit 1
fi
# Prompt the user for new values and update the file


read -p "Enter new value for SOLVER_COUNT ( 0:both 1:ksolver 2:MultSolver): " solver_count
update_define "SOLVER_COUNT" "$solver_count"
if [[ $solver_count == 0 ]]; then
    read -p "Enter new value for K-walk :: NO_OF_MSJ: " no_of_msj
    update_define "NO_OF_MSJ" "$no_of_msj"

    read -p "Enter new value for K-walk :: MAX_EPOCH: " max_epoch
    update_define "MAX_EPOCH" "$max_epoch"

    read -p "Enter new value for MM ::REPEAT_MULT: " repeat_mult
    update_define "REPEAT_MULT" "$repeat_mult"

    read -p "Run for residual (0:no and 1:yes)" residual_run
    update_define "RESIDUAL_RUN" "$residual_run"
    if [[ $residual_run == 1 ]]; then
        read -p "Residual Threshold(eg: 0.001)" residual_th
        update_define "RESIDUAL_THRESHOLD" "$residual_th"
    fi

elif [[ $solver_count == 1 ]]; then
    read -p "Enter new value for K-walk :: NO_OF_MSJ: " no_of_msj
    update_define "NO_OF_MSJ" "$no_of_msj"

    # read -p "Enter new value for K-walk :: MAX_EPOCH: " max_epoch
    # update_define "MAX_EPOCH" "$max_epoch"

    read -p "Run for residual (0:no and 1:yes)" residual_run
    update_define "RESIDUAL_RUN" "$residual_run"
    if [[ $residual_run == 1 ]]; then
        read -p "Residual Threshold(eg: 0.001)" residual_th
        update_define "RESIDUAL_THRESHOLD" "$residual_th"
    fi

elif [[ $solver_count == 2 ]]; then
    read -p "Enter new value for MM ::REPEAT_MULT: " repeat_mult
    update_define "REPEAT_MULT" "$repeat_mult"

    read -p "Run for residual (0:no and 1:yes)" residual_run
    update_define "RESIDUAL_RUN" "$residual_run"
    if [[ $residual_run == 1 ]]; then
        read -p "Residual Threshold(eg: 0.001)" residual_th
        update_define "RESIDUAL_THRESHOLD" "$residual_th"
    fi
else
    echo "Unknown solver: $solver_count"
    exit 1
fi
echo "Please enter the new values for the configuration file."
update_define "INPUTpmAT" "\"$NEW_INPUT_PMAT_PATH\""
update_define "INPUTgRAPH" "\"$NEW_INPUT_GRAPH_PATH\""
update_define "OpFileSolK" "\"${NEW_OPFILESOLK_PATH}${no_of_msj}\"" 
update_define "OpFileSolM" "\"${NEW_OPFILESOLM_PATH}${repeat_mult}\""



echo "Configuration updated successfully in $CONFIG_FILE."
 
cd ..
make
./bin/main_app
# mv ./bin/main_app ./bin/${DATASET_NAME}${no_of_msj}

