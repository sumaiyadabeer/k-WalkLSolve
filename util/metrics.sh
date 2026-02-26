#!/bin/bash
folder_path="/home/sumaiya/FixedPS/kSolve/data/outputs/gene"
centrality_m="${folder_path}/eta/gene_m-7_eta_b"
centrality_k="${folder_path}/log_4/gene_k131072"
rank="_rank"
relay_rank="_relay_rank"
scatter_plt_file="${centrality_k}_scatter.png"
scatter_plt_relay_file="${centrality_k}_scatter_relay.png"
# difference_plt_file="/home/sumaiya/FixedPS/kSolve/data/outputs/gene/gene_diff.png"

python full_rank.py --mRank ${centrality_m}${relay_rank} --kRank ${centrality_k}${relay_rank} --sctrPlt $scatter_plt_relay_file 

# # source not ignored
# echo "source vertices considered"
# python rank_creation.py --mFile $centrality_m --kFile $centrality_k --ignrSrc "False"
# python full_rank.py --mRank ${centrality_m}${rank} --kRank ${centrality_k}${rank} --sctrPlt $scatter_plt_file 
# python partial_rank.py --mRank ${centrality_m}${rank} --kRank ${centrality_k}${rank} 

# echo "source vertices ignored"
# # source ignored 
# python rank_creation.py --mFile $centrality_m --kFile $centrality_k --ignrSrc "True"
# python full_rank.py --mRank ${centrality_m}${relay_rank} --kRank ${centrality_k}${relay_rank} --sctrPlt $scatter_plt_relay_file 
# python partial_rank.py --mRank ${centrality_m}${relay_rank} --kRank ${centrality_k}${relay_rank} 


echo "bash script is complete"