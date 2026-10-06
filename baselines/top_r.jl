using Printf

function extract_top_r(cnt::AbstractVector{T}, r::Int) where T
    V = length(cnt)
    r_actual = min(r, V)
    
    # partialsortperm is an O(V log r) algorithm native to Julia
    top_indices_actual = partialsortperm(cnt, 1:r_actual, rev=true)
    top_values_actual = cnt[top_indices_actual]
    
    # Pad with -1 to match the C++ logic exactly if r > V
    if r_actual < r
        top_indices = fill(-1, r)
        top_values = fill(T(-1), r) 
        top_indices[1:r_actual] .= top_indices_actual
        top_values[1:r_actual] .= top_values_actual
        return top_indices, top_values
    end
    
    return top_indices_actual, top_values_actual
end

# function evaluate_round_metrics(round_num::Int, r::Int, 
#                                 curr_idx::AbstractVector{Int}, curr_eta::AbstractVector{Float32}, 
#                                 prev_idx::AbstractVector{Int}, prev_eta::AbstractVector{Float32})
#     println("\n==================================================")
#     println(" METRICS COMPARISON FOR ROUND $round_num (Top R = $r)")
#     println("==================================================")

#     # 1. Build lookup dictionary (Dicts are highly optimized in Julia)
#     prev_relevance_map = Dict{Int, Float32}(zip(prev_idx, prev_eta))

#     # 2. Calculate DCG 
#     # Utilizing a generator avoids allocating an intermediate array
#     dcg = sum(get(prev_relevance_map, curr_idx[i], 0.0f0) / log2(Float32(i + 1)) for i in 1:r)

#     # 3. Calculate Ideal DCG
#     ideal_relevance = sort(prev_eta, rev=true)
#     r_ideal = min(r, length(ideal_relevance))
#     idcg = sum(ideal_relevance[i] / log2(Float32(i + 1)) for i in 1:r_ideal)

#     ndcg = idcg > 0.0f0 ? dcg / idcg : 0.0f0

#     @printf(" -> Graded NDCG:     %.6f\n", ndcg)
#     println("==================================================")

#     return ndcg
# end

# using Printf

function evaluate_round_metrics(round_num::Integer, r::Integer, 
                                curr_idx::AbstractVector{<:Integer}, curr_eta::AbstractVector{<:Real}, 
                                prev_idx::AbstractVector{<:Integer}, prev_eta::AbstractVector{<:Real})
    println("\n==================================================")
    println(" METRICS COMPARISON FOR ROUND $round_num (Top R = $r)")
    println("==================================================")

    # 1. Build lookup dictionary (Letting Julia infer types automatically via zip)
    prev_relevance_map = Dict(zip(prev_idx, prev_eta))
    
    # Extract the numeric type of eta (e.g., Float64) so we can use a type-stable zero
    T = eltype(curr_eta)

    # 2. Calculate DCG 
    # Utilizing a generator avoids allocating an intermediate array
    dcg = sum(get(prev_relevance_map, curr_idx[i], zero(T)) / log2(i + 1) for i in 1:r)

    # 3. Calculate Ideal DCG
    ideal_relevance = sort(prev_eta, rev=true)
    r_ideal = min(r, length(ideal_relevance))
    idcg = sum(ideal_relevance[i] / log2(i + 1) for i in 1:r_ideal)

    # Calculate final NDCG
    ndcg = idcg > zero(T) ? dcg / idcg : zero(T)

    # @printf requires a specific type, so we safely cast to Float64 for formatting
    @printf(" -> Graded NDCG:     %.6f\n", Float64(ndcg))
    println("==================================================")

    return ndcg
end