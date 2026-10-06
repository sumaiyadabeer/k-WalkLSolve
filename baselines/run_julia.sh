cd /home/cse/phd/csz208509/k-WalkLSolve-master/baselines
# Clear library path conflicts and set headless environment
export LD_LIBRARY_PATH=""
export GKSwstype=100
export DISPLAY=""

# echo "wikibug: "
# timeout -k 200s 7000s ~/julia-1.10.8/bin/julia solvers.jl /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/wiki-bug/wiki_bug_ijv /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/wiki-bug/b_sink.txt /home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/wiki-bug/ && echo OK || echo Failed, status $? 

# echo "wikioc: "
# timeout -k 200s 7000s ~/julia-1.10.8/bin/julia solvers.jl /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/wiki_oc/wiki_oc_ijv /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/wiki_oc/b_sink.txt /home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/wiki_oc/ && echo OK || echo Failed, status $? 

# echo "gene: "
# timeout -k 200s 7000s ~/julia-1.10.8/bin/julia solvers.jl /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/gene/gene_ijv /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/gene/b_sink.txt /home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/gene/ && echo OK || echo Failed, status $? 



# echo "holl09:"
# timeout -k 200s 7000s ~/julia-1.10.8/bin/julia solvers.jl /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/holl09/holl09_ijv /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/holl09/b_sink.txt /home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/holl09/ && echo OK || echo Failed, status $? 

# echo "holl11:"
# timeout -k 200s 7000s ~/julia-1.10.8/bin/julia solvers.jl /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/holl11/holl11_ijv /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/holl11/b_sink.txt /home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/holl11/ && echo OK || echo Failed, status $? 

echo "asSki:"
timeout -k 200s 7000s ~/julia-1.10.8/bin/julia solvers.jl /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/as_ski/as_ski_ijv /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/as_ski/b_sink.txt /home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/as_ski/ && echo OK || echo Failed, status $? 

# echo "brain: "
# timeout -k 200s 7000s ~/julia-1.10.8/bin/julia solvers.jl /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/brain/brain_ijv /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/brain/b_sink.txt /home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/brain/ && echo OK || echo Failed, status $? 

echo "wikisr:"
timeout -k 200s 7000s ~/julia-1.10.8/bin/julia solvers.jl /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/wiki-sr/wiki_sr_ijv /home/cse/phd/csz208509/k-WalkLSolve-master/data/datasets/wiki-sr/b_sink.txt /home/cse/phd/csz208509/k-WalkLSolve-master/data/outputs/wiki-sr/ && echo OK || echo Failed, status $? 

