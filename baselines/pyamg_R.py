# ------------------------------------------------------------------
# Step 1: import scipy and pyamg packages
# ------------------------------------------------------------------
import numpy as np
import pyamg
import matplotlib.pyplot as plt

import scipy
import time
import fileinput
import argparse

# python3 pyamg_demo.py -csr /home/sumaiya/Cuda_solver_optimization1/data/wiki_bug/wikipedia_link_bug_csr -pyamFile /home/sumaiya/Cuda_solver_optimization1/data/wiki_bug/pyamg.out


# can use directly 
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
# csr_matrix((values, col_off, row_ptr), shape= (len(row_ptr)-1, len(row_ptr)-1)).toarray()
# LOL
# scipy.sparse.csgraph.laplacian


def LapMat_frm_CSR(row_ptr, col_off, values):

    A = csr_matrix((values, col_off, row_ptr), shape= (len(row_ptr)-1, len(row_ptr)-1))

    # N = len(row_ptr)-1
#     A = [0]*(N*N)
#     for i in range(0,len(row_ptr)-1):
#         degree = row_ptr[i+1]-row_ptr[i]
#         A[(N*i)+i] = degree
# #         print(N,i, N*i)
#         for j in range(row_ptr[i],row_ptr[i+1]):
# #             print(i,j)
#             A[(col_off[j])+N*(i)]= -values[j]


#     A=np.asarray(A)
#     # reshape array into 4 rows x 2 columns, and transpose the result
    # A = A.reshape(N, N).T 
    return A

def read_graph(filename):
    
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    file1.close()

    node_edge = Lines[0]
    node_edge = node_edge.split()
    node_edge = [int(i) for i in node_edge]
    
    row_ptr = Lines[1]
    row_ptr = row_ptr.split()
    row_ptr = [int(i) for i in row_ptr]
    
    col_off = Lines[2]
    col_off = col_off.split()
    col_off = [int(i) for i in col_off]
    
    values = Lines[3]
    values = values.split()
    values = [float(i) for i in values]
    
    b = Lines[4]
    b = b.split()
    b = [int(i) for i in b]
    
    
    return node_edge, row_ptr, col_off, values, b

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
# parser.add_argument("-lapFile", "--lapFile", help = "Name of Laplacian output file")
parser.add_argument("-csr", "--csr", help = "Name of input file")
parser.add_argument("-pyamFile", "--pyamFile", help = "Name of Pyamg output file")
 
# Read arguments from command line
args = parser.parse_args()

# print("Started reading CSR file")
node_edge, row_ptr, col_off, values, b = read_graph(args.csr)
# print("Started creating Laplacian matrix")
L = LapMat_frm_CSR(row_ptr, col_off, values)


dimension = len(row_ptr)-1
sink_index = [i for i in range(0, len(b)) if b[i] < 0]
assert(len(sink_index)==1) #to make sure only one elecemt is negative 
sink_index = sink_index[0]

# L = np.asmatrix(L)
# degrees = np.diag(L)
# print(degrees)
# np.savetxt( "web_oc_degrees.txt",  [degrees], delimiter='\n')


# print("Creating sparse a and np.array b")
a = csgraph.laplacian(L)
a = csr_matrix(a)
# print(type(a))
# a = scipy.sparse.csr_matrix(L)

b = np.array(b)
# b = b/(-b[sink_index])
# beta = 0.125 
# b = beta*b

# print("Runing the PyAMG default")
# ------------------------------------------------------------------
# Step 2: setup up the system using pyamg.gallery
# ------------------------------------------------------------------
n = dimension

# A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
# b = np.random.rand(A.shape[0])                     # pick a random right hand side


res1 = []
t1_ = time.time()
# X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
# stencil = pyamg.gallery.diffusion_stencil_2d(type='FE', epsilon=0.001, theta=np.pi / 3)

ml = pyamg.ruge_stuben_solver(a)
# ml2 = pyamg.smoothed_aggregation_solver(a)
# print("Solver is designed successfully")
x = ml.solve(b, tol=10, residuals=res1)  # solve Ax=b to a tolerance of 1e-12
time_taken1 =  time.time() - t1_

# convert x to x_sink

# print("Calculating residual of PyAMG")
x = x - x[sink_index]

residual = b - a * x
# residual[sink_index] = 0

# for line_no, line in enumerate(fileinput.FileInput(args.lapFile, inplace=1)):
#     if line_no ==  1:
#         print(np.linalg.norm(residual_x_sink_zero))
#     if line_no == 0: 
#         print(time_taken)
# print("writing results for pyamg")
with open(args.pyamFile, 'w') as f:
    # f.writelines(str(time_taken1) + '\n')
    # f.writelines(str(np.linalg.norm(residual)) + '\n')
    np.savetxt(f,  [x], delimiter='\n')

print("Pyamg_R")
print("Time",time_taken1, "\t residual", np.linalg.norm(residual/np.linalg.norm(b)))

# #  Write time residual and x

# # exit()

# # print("Runing the PyAMG Smooth Aggregration")
# # ------------------------------------------------------------------
# # Step 2: setup up the system using pyamg.gallery
# # ------------------------------------------------------------------
# n = dimension

# # A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
# # b = np.random.rand(A.shape[0])                     # pick a random right hand side


# res1 = []
# t_ = time.time()
# # X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
# # stencil = pyamg.gallery.diffusion_stencil_2d(type='FE', epsilon=0.001, theta=np.pi / 3)

# # ml = pyamg.ruge_stuben_solver(a)
# ml = pyamg.smoothed_aggregation_solver(a)

# x = ml.solve(b, tol=10, residuals=res1)  # solve Ax=b to a tolerance of 1e-12
# time_taken =  time.time() - t_

# # convert x to x_sink

# # print("Calculating residual of PyAMG Smooth Aggregration")
# x = x - x[sink_index]

# residual = b - a * x
# # residual[sink_index] = 0

# # for line_no, line in enumerate(fileinput.FileInput(args.lapFile, inplace=1)):
# #     if line_no ==  1:
# #         print(np.linalg.norm(residual_x_sink_zero))
# #     if line_no == 0: 
# #         print(time_taken)
# # print("writing results for pyamg Smooth Aggregration")
# with open(args.pyamFile+'SA', 'w') as f:
#     # f.writelines(str(time_taken) + '\n')
#     # f.writelines(str(np.linalg.norm(residual)) + '\n')
#     np.savetxt(f,  [x], delimiter='\n')

# print("PyamgSA")
# print("Time",time_taken, "\t Relative Residual", np.linalg.norm(residual)/np.linalg.norm(b))
