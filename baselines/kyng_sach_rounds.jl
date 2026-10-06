using Laplacians
using SparseArrays
using LinearAlgebra
using DelimitedFiles

if length(ARGS) < 4
    println("Usage: julia kyng_sach_solver.jl <ijv_file> <b_file> <output_file> <num_fixed_iterations>")
    exit(1)
end

ijv_file = ARGS[1]
b_file = ARGS[2]
output_file = ARGS[3]
num_fixed_its = parse(Int, ARGS[4])

# --- 1. Load Graph (IJV Format) ---
println("Reading input graph...")
ijv = readdlm(ijv_file)
ijv = convert(Matrix{Int64}, ijv)

i = ijv[2:end, 1]
j = ijv[2:end, 2]
e = ijv[1, 2]

loops = Vector{UInt64}()
for index in 1:length(i)
    if i[index] == j[index]
        push!(loops, index)
    end
end

i_full = [i; j]
j_full = [j; i]
v = ones(Float64, length(i_full))

a = sparse(i_full, j_full, v)

for index in loops
    a[index, index] = 0.0
end

a = sparse(Float64.(a))
dropzeros!(a)
la = lap(a)
degrees = diag(la)

# --- 2. Load Vector b and Sink Index ---
println("Reading vector b...")
b_data = readdlm(b_file)
b = Vector{Float64}(b_data[1, :])

no_of_sources = -minimum(b)
sink_index = findfirst(==(-no_of_sources), b)

# --- 3. NDCG@50 Calculation Function ---
function compute_consecutive_ndcg(prev_Dx::AbstractVector{Float64}, curr_Dx::AbstractVector{Float64}, r::Int=50)
    p_vec = Vector(prev_Dx)
    c_vec = Vector(curr_Dx)
    
    n_nodes = length(p_vec)
    r = min(r, n_nodes)
    
    # Ranking order based on previous round
    ideal_perm = sortperm(p_vec, rev=true)[1:r]
    idcg = sum(p_vec[ideal_perm[k]] / log2(k + 1) for k in 1:r)
    
    if idcg == 0.0
        return 1.0
    end
    
    # Current round ranking evaluated against previous scores
    curr_perm = sortperm(c_vec, rev=true)[1:r]
    dcg = sum(p_vec[curr_perm[k]] / log2(k + 1) for k in 1:r)
    
    return dcg / idcg
end

# --- 4. Iterative Round Solver ---
prev_Dx = Float64[]
max_rounds = 100
ndcg_threshold = 0.95

t_start = time()

for round in 1:max_rounds
    global prev_Dx
    target_iterations = round * num_fixed_its
    
    # Configure solver for cumulative iterations
    solver = approxchol_lap(a, maxits=target_iterations, tol=0.0)
    
    iters_taken = zeros(Int, 1)
    x = solver(b, pcgIts=iters_taken)
    
    # Ground the sink and compute degree-multiplied solution D * x
    x_shifted = x .- x[sink_index]
    curr_Dx = x_shifted .* degrees
    
    res_norm = norm(la * x - b)
    
    if round == 1
        println("Round 1 completed | Iterations: $(iters_taken[1]) | Residual: $(res_norm)")
        prev_Dx = copy(curr_Dx)
    else
        ndcg_val = compute_consecutive_ndcg(prev_Dx, curr_Dx, 50)
        println("Round $(round) completed | Iterations: $(iters_taken[1]) | Residual: $(res_norm) | Consecutive NDCG@50: $(ndcg_val)")
        
        if ndcg_val > ndcg_threshold
            println("Convergence reached: Consecutive NDCG@50 > $(ndcg_threshold) at round $(round).")
            prev_Dx = copy(curr_Dx)
            break
        end
        
        prev_Dx = copy(curr_Dx)
    end
end

elapsed_time = time() - t_start
println("Solver finished in $(elapsed_time) seconds.")

# --- 5. Export Grounded D * x Vector ---
println("Writing grounded D*x solution to: $(output_file)")
open(output_file, "w") do io
    writedlm(io, prev_Dx)
end
println("Output saved successfully.")