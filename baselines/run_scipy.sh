
module load anaconda/2025.06
source activate
conda activate python_solver
cd /home/cse/phd/csz208509/k-WalkLSolve-master/baselines
BASE_DIR="/home/cse/phd/csz208509/k-WalkLSolve-master/data"
echo "scipy_solver: bug"
python3 scipy_rounds.py  "${BASE_DIR}/datasets/wiki-bug/wiki_bug_csr_b-sink"  "${BASE_DIR}/outputs/wiki-bug/scipy" 4913
echo "scipy_solver: wikioc"
python3 scipy_rounds.py  "${BASE_DIR}/datasets/wiki_oc/wiki_oc_csr_b-sink" "${BASE_DIR}/outputs/wiki_oc/scipy" 729
echo "scipy_solver: wikisr" 
python3 scipy_rounds.py  "${BASE_DIR}/datasets/wiki-sr/wiki_sr_csr_b-sink" "${BASE_DIR}/outputs/wiki-sr/scipy" 1000
echo "scipy_solver: brain" 
python3 scipy_rounds.py  "${BASE_DIR}/datasets/brain/brain_csr_b-sink" "${BASE_DIR}/outputs/brain/scipy" 216
echo "scipy_solver: gene" 
python3 scipy_rounds.py  "${BASE_DIR}/datasets/gene/gene_csr_b-sink" "${BASE_DIR}/outputs/gene/scipy" 216
echo "scipy_solver: holl09" 
python3 scipy_rounds.py  "${BASE_DIR}/datasets/holl09/holl09_csr_b-sink" "${BASE_DIR}/outputs/holl09/scipy" 216
echo "scipy_solver: holl11" 
python3 scipy_rounds.py  "${BASE_DIR}/datasets/holl11/holl11_csr_b-sink" "${BASE_DIR}/outputs/holl11/scipy" 216
echo "scipy_solver: asSki" 
python3 scipy_rounds.py  "${BASE_DIR}/datasets/as_ski/as_ski_csr_b-sink" "${BASE_DIR}/outputs/as_ski/scipy" 15625

