# echo "pyamg_SA: wikioc"
# python3 pyamg_SA.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikioc/oc_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/wikioc/pyamg
# echo "pyamg_SA: wikisr" 
# python3 pyamg_SA.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikisr/out.wikipedia_link_sr_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/wikisr/pyamg
# echo "pyamg_SA: brain" 
# python3 pyamg_SA.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/brain/brain.edges_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/brain/pyamg
# echo "pyamg_SA: gene" 
# python3 pyamg_SA.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/gene/gene.edges_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/gene/pyamg
# echo "pyamg_SA: holl09" 
# python3 pyamg_SA.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-09/holl09_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/holl09/pyamg
# echo "pyamg_SA: holl11" 
# python3 pyamg_SA.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-11/holl11_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/holl11/pyamg
# echo "pyamg_SA: asSki" 
# python3 pyamg_SA.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/as-ski/as-skitter.txt_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/asSki/pyamg

# echo "pyamg_R: wikioc" more than 6 hours
# python3 pyamg_R.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikioc/oc_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/wikioc/pyamg
# echo "pyamg_R: wikisr" more than 6 hours
# python3 pyamg_R.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikisr/out.wikipedia_link_sr_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/wikisr/pyamg
echo "pyamg_R: gene" 
python3 pyamg_R.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/gene/gene.edges_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/gene/pyamg
echo "pyamg_R: brain" 
python3 pyamg_R.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/brain/brain.edges_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/brain/pyamg
echo "pyamg_R: holl09" 
python3 pyamg_R.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-09/holl09_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/holl09/pyamg
echo "pyamg_R: holl11" 
python3 pyamg_R.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/holl-11/holl11_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/holl11/pyamg
echo "pyamg_R: asSki" 
python3 pyamg_R.py -csr /home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/as-ski/as-skitter.txt_csr -pyamFile /home/sumaiya/FixedPS/kSolve/data/outputs/asSki/pyamg
