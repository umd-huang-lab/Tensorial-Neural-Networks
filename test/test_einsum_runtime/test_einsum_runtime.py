import torch
import time




I = 10000
J = 100000
X = torch.ones(I)
Y = torch.ones(J)

start = time.time()
A = torch.einsum("i,j->ij", X, Y)
finish_test1 = time.time()
B = torch.einsum("i,j->", X, Y)
finish_test2 = time.time()




print("Elapsed seconds test1 = " + str(finish_test1 - start))
print("Elapsed seconds test2 = " + str(finish_test2 - start))
