Gfilename="/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/as-ski/as-skitter.txt_csr"
op_k_path="/home/sumaiya/FixedPS/kSolve/data/outputs/asSki/log_4/asSki_k"
Msol="/home/sumaiya/FixedPS/kSolve/data/outputs/asSki/eta/asSki_m-7_eta_b"
#!/bin/bash 
s=("8192" "16384" "32768" "65536" "131072") 
centrlty="_centrality"
OpFileM=${Msol}${centrlty}

i="4096"
./cen $Gfilename ${op_k_path}${i} $Msol ${op_k_path}${i}${centrlty} $OpFileM 1
    for n in ${s[@]}; 
    do
        Ksol=${op_k_path}${n}
        OpFileK=${op_k_path}${n}${centrlty}

        ./cen $Gfilename $Ksol $Msol $OpFileK $OpFileM 0
        
        # echo $Ksol
        # echo $OpFileK
        # echo $Msol
        # echo $OpFileM
        # echo " "
    done

