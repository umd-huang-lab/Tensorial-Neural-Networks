import torch
import time



def i_j_k_to_ijk_ij_i__disjoint(A, B, C):
    X = torch.einsum("i,j,k->ijk", A, B, C)
    Y = torch.einsum("i,j,k->ij", A, B, C)
    Z = torch.einsum("i,j,k->i", A, B, C)
    W = torch.einsum("i,j,k->", A, B, C)

    return [X, Y, Z, W]

def i_j_k_to_ijk_ij_i__reuse(A, B, C):
    X = torch.einsum("i,j,k->ijk", A, B, C)
    Y = torch.einsum("ijk->ij", X)
    Z = torch.einsum("ij->i", Y)
    W = torch.einsum("i->", Z)

    return [X, Y, Z, W]


I = 1000
J = 1000
K = 100

A = torch.rand(I)
B = torch.rand(J)
C = torch.rand(K)

start = time.time()

#out_disjoint = i_j_k_to_ijk_ij_i__disjoint(A, B, C)
out_reuse = i_j_k_to_ijk_ij_i__reuse(A, B, C)

finish = time.time()


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





