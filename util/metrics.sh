#!/bin/bash
folder_path="/home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/as_ski"
centrality_m="${folder_path}/pcg_sol"
centrality_k="${folder_path}/asSki_262k262144"
rank="_rank"
relay_rank="_relay_rank"
scatter_plt_file="${centrality_k}_scatter.png"
scatter_plt_relay_file="${centrality_k}_scatter_relay.png"
# difference_plt_file="/home/sumaiya/FixedPS/kSolve/data/outputs/gene/gene_diff.png"


# source not ignored
echo "source vertices considered"
python rank_creation.py --mFile $centrality_m --kFile $centrality_k --ignrSrc "False"
python full_rank.py --mRank ${centrality_m}${rank} --kRank ${centrality_k}${rank} --sctrPlt $scatter_plt_file 
python partial_rank.py --mRank ${centrality_m}${rank} --kRank ${centrality_k}${rank} 


echo "bash script is complete"
