import torch
import time



#def i_j_k_to_ijk_ij_i__disjoint(A, B, C):
#    X = torch.einsum("i,j,k->ijk", A, B, C)
#    Y = torch.einsum("i,j,k->ij", A, B, C)
#    Z = torch.einsum("i,j,k->i", A, B, C)
#    W = torch.einsum("i,j,k->", A, B, C)
#
#    return [X, Y, Z, W]
#
#def i_j_k_to_ijk_ij_i__reuse(A, B, C):
#    X = torch.einsum("i,j,k->ijk", A, B, C)
#    Y = torch.einsum("ijk->ij", X)
#    Z = torch.einsum("ij->i", Y)
#    W = torch.einsum("i->", Z)
#
#    return [X, Y, Z, W]
#
#
#I = 1000
#J = 1000
#K = 100
#
#A = torch.rand(I)
#B = torch.rand(J)
#C = torch.rand(K)
#
#start = time.time()
#
#out_disjoint = i_j_k_to_ijk_ij_i__disjoint(A, B, C)
##out_reuse = i_j_k_to_ijk_ij_i__reuse(A, B, C)
#
#finish = time.time()
#


#def i_j_k_to_ij_i_disjoint(A, B, C):
#    X = torch.einsum("i,j,k->ij", A, B, C)
#    Y = torch.einsum("i,j,k->i", A, B, C)
#
#    return [X, Y]
#
#def i_j_k_to_ij_i_reuse(A, B, C):
#    X = torch.einsum("i,j,k->ij", A, B, C)
#    Y = torch.einsum("ij->i", X)
#
#    return [X, Y]
#
#def i_j_k_to_ij_i_intermediate(A, B, C):
#    INT = torch.einsum("i,j->ij", A, B)
#    X = torch.einsum("ij,k->ij", INT, C)
#    Y = torch.einsum("ij,k->i", INT, C)
#
#    return [X, Y]
#
#
#I = 10000
#J = 10000
#K = 1000
#
#A = torch.ones(I, device='cpu')
#B = torch.ones(J, device='cpu')
#C = torch.ones(K, device='cpu')
#
#start = time.time()
#
#out_disjoint = i_j_k_to_ij_i_disjoint(A, B, C)
#out_reuse = i_j_k_to_ij_i_reuse(A, B, C)
#out_intermediate = i_j_k_to_ij_i_intermediate(A, B, C)
#
#finish = time.time()
#






I = 10000
J = 10000
K = 1000

A = torch.rand(I)
B = torch.rand(J)
C = torch.rand(K)

start = time.time()

out_ij = torch.einsum("i,j,k->ij", A, B, C)
out_i = torch.einsum("i,j,k->i", A, B, C)

finish = time.time()




#I = 1000000
#J = 1000000
#A = torch.rand(I)
#B = torch.rand(J)
#
#def i_j_to_i__disjoint(A, B):
#    X = torch.einsum("i,j->i", A, B)
#    Y = torch.einsum("i,j->", A, B)
#
#    return [X, Y]
#
#
#def i_j_to_i__reuse(A, B):
#    X = torch.einsum("i,j->i", A, B)
#    Y = torch.einsum("i->", X)
#
#    return [X, Y]
#
#
#start = time.time()
#
##out_disjoint = i_j_to_i__disjoint(A, B)
#out_reuse = i_j_to_i__reuse(A, B)
#
#finish = time.time()



#I = 100000
#J = 10000
#X = torch.rand(I)
#Y = torch.rand(J)
#
#start = time.time()
#B = torch.einsum("i,j->", X, Y)
##A = torch.einsum("i,j->ij", X, Y)
#finish = time.time()
#


#I = 10000
#J = 100000
#X = torch.rand(I, J)
#
#
#start = time.time()
##A = torch.einsum("ij->", X)
##A = torch.einsum("ij,ij->", X, X)
#
#A = torch.einsum("ij,ij->ij", X, X)
#B = torch.einsum("ij->", A)
#
#finish = time.time()
#


print("Elapsed seconds test = " + str(finish - start))





