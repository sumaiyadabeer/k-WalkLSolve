module load cuda/13.0
cd /home/cse/phd/csz208509/k-WalkLSolve-master/baselines/GLS/
make
BASE_DIR="/home/cse/phd/csz208509/k-WalkLSolve-master/data"
echo "GLS_solver: bug"
timeout -k 200s 7000s ./main "${BASE_DIR}/datasets/wiki-bug/wiki_bug_csr_b-sink" "${BASE_DIR}/outputs/wiki-bug/GLS" 4913
echo "GLS_solver: wikioc"
timeout -k 200s 7000s ./main "${BASE_DIR}/datasets/wiki_oc/wiki_oc_csr_b-sink" "${BASE_DIR}/outputs/wiki_oc/GLS" 729
echo "GLS_solver: wikisr" 
timeout -k 200s 7000s ./main "${BASE_DIR}/datasets/wiki-sr/wiki_sr_csr_b-sink" "${BASE_DIR}/outputs/wiki-sr/GLS" 1000
echo "GLS_solver: brain" 
timeout -k 200s 7000s ./main "${BASE_DIR}/datasets/brain/brain_csr_b-sink" "${BASE_DIR}/outputs/brain/GLS" 216
echo "GLS_solver: gene" 
timeout -k 200s 7000s ./main "${BASE_DIR}/datasets/gene/gene_csr_b-sink" "${BASE_DIR}/outputs/gene/GLS" 216
echo "GLS_solver: holl09" 
timeout -k 200s 7000s ./main "${BASE_DIR}/datasets/holl09/holl09_csr_b-sink" "${BASE_DIR}/outputs/holl09/GLS" 216
echo "GLS_solver: holl11" 
timeout -k 200s 7000s ./main "${BASE_DIR}/datasets/holl11/holl11_csr_b-sink" "${BASE_DIR}/outputs/holl11/GLS" 216
echo "GLS_solver: asSki" 
timeout -k 200s 7000s ./main "${BASE_DIR}/datasets/as_ski/as_ski_csr_b-sink" "${BASE_DIR}/outputs/as_ski/GLS" 15625


