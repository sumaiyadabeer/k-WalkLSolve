#!/bin/bash

folder_path="/home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs"

# Array containing the 8 centrality_m file basenames
centrality_m_files=(
    "wiki-bug/wikibug_b11_pcg_sol"
    "wiki_oc/wikioc_b11_pcg_sol"
    "wiki-sr/wikisr_b11_pcg_sol" 
    "brain/brain_b11_pcg_sol" 
    "gene/gene_b11_pcg_sol" 
    "holl09/holl09_b11_pcg_sol"
    "holl11/holl11_b11_pcg_sol"
    "as_ski/asSki_b11_pcg_sol"    
)

# Array containing the 8 matching centrality_k file basenames
centrality_k_files=(
    "wiki-bug/wikibug_b11_262144" 
    "wiki_oc/wikioc_b11_262144"    
    "wiki-sr/wikisr_b11_262144"    
    "brain/brain_b11_262144"     
    "gene/gene_b11_262144"     
    "holl09/holl09_b11_262144"    
    "holl11/holl11_b11_262144"    
    "as_ski/asSki_b11_262144"
)

rank="_rank"
relay_rank="_relay_rank"

# Loop over the indices 0 to 7
for i in "${!centrality_m_files[@]}"; do
    centrality_m="${folder_path}/${centrality_m_files[$i]}"
    centrality_k="${folder_path}/${centrality_k_files[$i]}"
    
    scatter_plt_file="${centrality_k}_scatter.png"
    scatter_plt_relay_file="${centrality_k}_scatter_relay.png"

    echo "--------------------------------------------------"
    echo "Processing pair $((i+1)): ${centrality_m_files[$i]} and ${centrality_k_files[$i]}"
    echo "source vertices considered"

    python rank_creation.py --mFile "$centrality_m" --kFile "$centrality_k" --ignrSrc "False"
    python full_rank.py --mRank "${centrality_m}${rank}" --kRank "${centrality_k}${rank}" --sctrPlt "$scatter_plt_file" 
    python partial_rank.py --mRank "${centrality_m}${rank}" --kRank "${centrality_k}${rank}"
done

echo "bash script is complete"
