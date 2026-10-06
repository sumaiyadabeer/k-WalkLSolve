
module load anaconda/2025.06
source activate
conda activate python_solver
cd /home/cse/phd/csz208509/k-WalkLSolve-master/baselines
BASE_DIR="/home/cse/phd/csz208509/k-WalkLSolve-master/data"
# echo "pyaSA_solver: bug"
# timeout -k 200s 7000s  python3pyaSA_rounds.py  -csr "${BASE_DIR}/datasets/wiki-bug/wiki_bug_csr_b-sink"  -pyamFile "${BASE_DIR}/outputs/wiki-bug/pyaSA" -num_iter  4913
# echo "pyaSA_solver: wikioc"
# timeout -k 200s 7000s  python3pyaSA_rounds.py  -csr "${BASE_DIR}/datasets/wiki_oc/wiki_oc_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/wiki_oc/pyaSA" -num_iter  729
# echo "pyaSA_solver: wikisr" 
# timeout -k 200s 7000s  python3pyaSA_rounds.py  -csr "${BASE_DIR}/datasets/wiki-sr/wiki_sr_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/wiki-sr/pyaSA" -num_iter  1000
# echo "pyaSA_solver: brain" 
# timeout -k 200s 7000s  python3pyaSA_rounds.py  -csr "${BASE_DIR}/datasets/brain/brain_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/brain/pyaSA" -num_iter  216
# echo "pyaSA_solver: gene" 
# timeout -k 200s 7000s  python3pyaSA_rounds.py  -csr "${BASE_DIR}/datasets/gene/gene_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/gene/pyaSA" -num_iter  216
# echo "pyaSA_solver: holl09" 
# timeout -k 200s 7000s  python3pyaSA_rounds.py  -csr "${BASE_DIR}/datasets/holl09/holl09_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/holl09/pyaSA" -num_iter  216
# echo "pyaSA_solver: holl11" 
# timeout -k 200s 7000s  python3pyaSA_rounds.py  -csr "${BASE_DIR}/datasets/holl11/holl11_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/holl11/pyaSA" -num_iter  216
# echo "pyaSA_solver: asSki" 
# timeout -k 200s 7000s  python3pyaSA_rounds.py  -csr "${BASE_DIR}/datasets/as_ski/as_ski_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/as_ski/pyaSA" -num_iter  15625

# echo "pyaR_solver: bug"
# timeout -k 200s 7000s  python3 pyaR_rounds.py  -csr "${BASE_DIR}/datasets/wiki-bug/wiki_bug_csr_b-sink"  -pyamFile "${BASE_DIR}/outputs/wiki-bug/ pyaR" -num_iter  4913
# echo "pyaR_solver: wikioc"
# timeout -k 200s 7000s  python3 pyaR_rounds.py  -csr "${BASE_DIR}/datasets/wiki_oc/wiki_oc_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/wiki_oc/ pyaR" -num_iter  729
# echo "pyaR_solver: wikisr" 
# timeout -k 200s 7000s  python3 pyaR_rounds.py  -csr "${BASE_DIR}/datasets/wiki-sr/wiki_sr_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/wiki-sr/ pyaR" -num_iter  1000
echo "pyaR_solver: gene" 
timeout -k 200s 7000s  python3 pyaR_rounds.py  -csr "${BASE_DIR}/datasets/gene/gene_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/gene/ pyaR" -num_iter  216
echo "pyaR_solver: holl09" 
timeout -k 200s 7000s  python3 pyaR_rounds.py  -csr "${BASE_DIR}/datasets/holl09/holl09_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/holl09/ pyaR" -num_iter  216
echo "pyaR_solver: holl11" 
timeout -k 200s 7000s  python3 pyaR_rounds.py  -csr "${BASE_DIR}/datasets/holl11/holl11_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/holl11/ pyaR" -num_iter  216
echo "pyaR_solver: brain" 
timeout -k 200s 7000s  python3 pyaR_rounds.py  -csr "${BASE_DIR}/datasets/brain/brain_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/brain/ pyaR" -num_iter  216
echo "pyaR_solver: asSki" 
timeout -k 200s 7000s  python3 pyaR_rounds.py  -csr "${BASE_DIR}/datasets/as_ski/as_ski_csr_b-sink" -pyamFile "${BASE_DIR}/outputs/as_ski/ pyaR" -num_iter  15625
