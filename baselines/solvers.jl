# VERSION
# using Pkg
# Pkg.activate("/Users/spielman/Lap7/")
import Pkg
#; Pkg.add("Plots")
using Laplacians
using Laplacians
using Laplacians
# using PyPlot
using Plots 
using SparseArrays
using Random
using LinearAlgebra
using Statistics
using Arpack 
using SparseArrays
using DelimitedFiles
# using ArgParse

#julia solvers.jl /home/sumaiya/Desktop/dataset/Dec/out.wikipedia_link_sr_ijv /home/sumaiya/Desktop/dataset/Dec/out.wikipedia_link_sr-b out.wikipedia_link_sr_out

println("Reading input is started")
ijv = readdlm(ARGS[1])

ijv = convert(Matrix{Int64}, ijv)

#creation of IJV n and e
i = ijv[:,1]
j = ijv[:,2]

n = i[1]
e = j[1]

i = i[2:e+1]
j = j[2:e+1]


loops = Vector{UInt64}()
for index in (length(i))
   if i[index] == j[index]
        append!(loops, index)
   end
end


# Adding j to i add for each i to j
i = [i; j]
j = [j; i[1:e]]
# Adding the weight of each edge
v = ones(2e,1)
v = v[:,1]

println("Reading input is done")
#creation of sparse matrix
a = sparse(i,j,v)

for index in loops
    a[index, index] = 0
end


a = sparse(a)
#match no of non zero enteries with row_ptr[n+1] in csr_conversion
la = lap(a) 
degrees = diag(la)

println("Creation of Adj and Sparse Matr is done")
# checking the degree of graph
## and creating the b

b = readdlm(ARGS[2])
b = convert(Matrix{Int64}, b)

#creation of IJV n and e
b = b[1,:]

no_of_sources = -minimum(b)
# beta =  3.125e-02
sink_index = findfirst(==(-no_of_sources), b)
# b = b ./ no_of_sources
# b = b .* beta
println("Reading of b is done")


# # recalculating norm for Lsolve
# x = readdlm("/home/sumaiya/Desktop/Datasets/julia_solvers/wiki_bug/results/b50_Lsolve")
# X = convert(Matrix{Float64}, x)

# # x = x[1,:]
# # x = x[3:14268,]
# print(length(x))
# residual = norm(la*x - b)
# print(residual)
# exit()

a = sparse(Float64.(a))
b = Float64.(b)

println("========== Matrix Diagnostics ==========")
println("size(a)          = ", size(a))
println("nnz(a)           = ", nnz(a))
println("eltype(a)        = ", eltype(a))
println("symmetric error  = ", norm(a - a'))
println("min(nonzeros(a)) = ", minimum(nonzeros(a)))
println("max(nonzeros(a)) = ", maximum(nonzeros(a)))
println("finite values    = ", all(isfinite, nonzeros(a)))
println("========================================")

# Remove self-loops, if any
a[diagind(a)] .= 0
dropzeros!(a)

# Check symmetry
if norm(a - a') > 1e-12
    error("Adjacency matrix is not symmetric.")
end

# Check weights
if !all(isfinite, nonzeros(a))
    error("Adjacency matrix contains NaN or Inf.")
end

if minimum(nonzeros(a)) < 0
    error("Adjacency matrix contains negative edge weights.")
end

 println("Cholesky 23 solver")
 # Spielman23 approxchol_lap: a fast heuristic for solving Laplacians equations written by Yuan Gao, Rasmus Kyng, and Daniel Spielman. It is described in the paper "Robust and Practical Solution of Laplacian Equations by Approximate Elimination" (https://arxiv.org/abs/2303.00709). 
 # sort(b) # chk the pattern again
 # tolerance does not matter in this case
 t = time()
  chol_solver = chol_lap(a) #recall this routine should pass adjacency matrix
  x = chol_solver(b)
  dt = time() - t
  shift = -x[sink_index]
  x = x .+ shift
  residual = la*x - b
  curr_Dx = x .* degrees
  # residual[sink_index] = 0
  # residual = norm(residual)



  filename = ARGS[3] * "_Chol_23"
  # FileIOStream =  open(filename,"w")
  # writedlm(FileIOStream, dt, "\n")
  # close(FileIOStream)
  print("Time: ", dt)

  # FileIOStream =  open(filename,"a")
  # writedlm(FileIOStream, residual, "\n")
  # close(FileIOStream)
  println("\t \t Residual: ", norm(residual))
  FileIOStream =  open(filename,"a")
  writedlm(FileIOStream, curr_Dx, "    ")
  close(FileIOStream)
  exit()

#measure time and write time residual and x 

# for i in 1:14266
#     print(a[i,i], "\n")
# end

# ?approxchol_lap
# approxchol_lap approxchol_sddm approxCholLap approxCholSDDM
# ParseError: KaTeX parse error: No such environment: verbatim at position 7: \begin{̲v̲e̲r̲b̲a̲t̲i̲m̲}̲ solver = appro…
# A heuristic by Daniel Spielman inspired by the linear system solver in https://arxiv.org/abs/1605.02353 by Rasmus Kyng and Sushant Sachdeva. Whereas that paper eliminates vertices one at a time, this eliminates edges one at a time. It is probably possible to analyze it. The \texttt{ApproxCholParams} let you choose one of three orderings to perform the elimination.

# ?chol_lap


print("Kyng Sach Solver")
t = time()
Kyng_Sach_solver = approxchol_lap(a, tol=0.1);
x = Kyng_Sach_solver(b)
dt = time() - t
shift = -x[sink_index]
x = x .+ shift
residual = la*x - b
# residual[sink_index] = 0
residual = norm(residual)

filename = ARGS[3] * "_Kyng_Sach"
# FileIOStream =  open(filename,"w")
# writedlm(FileIOStream, dt, "\n")
# close(FileIOStream)
print("Time: ", dt)

# FileIOStream =  open(filename,"a")
# writedlm(FileIOStream, residual, "\n")
# close(FileIOStream)
println("\t \t Residual: ", norm(residual))

FileIOStream =  open(filename,"a")
writedlm(FileIOStream, x, "    ")
close(FileIOStream)

# # # measure time and write time residual and x 

# # CMG solver using matlab
# # This is taking a while to connect
# include(string(Pkg.dir("Laplacians") , "/src/matlabSolvers.jl"))
# using MATLAB
# # ?matlab_ichol_lap
# println("CMG solver")
# t = time()
# cmg_solver =  matlabCmgLap(a; tol=0.1, maxits=10000)
# x = cmg_solver(b)
# dt = time() - t
# shift = -x[sink_index]
# x = x .+ shift
# residual = la*x - b
# # residual[sink_index] = 0
# residual = norm(residual)

# filename = ARGS[3] * "_CMG"
# # FileIOStream =  open(filename,"w")
# # writedlm(FileIOStream, dt, "\n")
# # close(FileIOStream)
# print("Time: ", dt)

# # FileIOStream =  open(filename,"a")
# # writedlm(FileIOStream, residual, "\n")
# # close(FileIOStream)
# println("\t \t Residual: ", residual)

# FileIOStream =  open(filename,"a")
# writedlm(FileIOStream, x, "    ")
# close(FileIOStream)

# # #measure time and write time residual and x 

# # sort(x)
 
