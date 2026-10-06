module load cuda/13.0
cd /home/cse/phd/csz208509/k-WalkLSolve-master/util
chmod +x custom_run_various_b.sh
echo "wikibug"
./custom_run_various_b.sh <<< "0"$'\n'"wikibug"$'\n'"262144"
echo "wikioc"
./custom_run_various_b.sh <<< "0"$'\n'"wikioc"$'\n'"262144"
echo "wikisr"
./custom_run_various_b.sh <<< "0"$'\n'"wikisr"$'\n'"262144"
echo "gene"
./custom_run_various_b.sh <<< "0"$'\n'"gene"$'\n'"262144"
echo "brain"
./custom_run_various_b.sh <<< "0"$'\n'"brain"$'\n'"262144"
echo "holl09"
./custom_run_various_b.sh <<< "0"$'\n'"holl09"$'\n'"262144"
echo "holl11"
./custom_run_various_b.sh <<< "0"$'\n'"holl11"$'\n'"262144"
echo "asSki"
./custom_run_various_b.sh <<< "0"$'\n'"asSki"$'\n'"262144"
