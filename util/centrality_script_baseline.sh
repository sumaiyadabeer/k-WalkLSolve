Gfilename="/home/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/as-ski/as-skitter.txt_csr"
baseline_sol_path="/home/sumaiya/FixedPS/kSolve/data/outputs/asSki/baselines/"

s=("Chol_23" "Kyng_Sach" "scipy_sol")
# s=("pyamg_SA" "pyamg" "Chol_23" "Kyng_Sach" "scipy_sol")
 for n in ${s[@]}; 
    do
        echo "./cen_baseline $Gfilename ${baseline_sol_path}${n}"
        ./cen_baseline $Gfilename ${baseline_sol_path}${n}
    done
