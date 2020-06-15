import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import tnn 

import torch

X = 128
Y = 512

R = 8
S = 3
T = 5
H = 1024
W = 1024


U = torch.rand(X, Y, S, device='cuda')

K = torch.rand(H, W, S, T, device='cuda')
# obtained by cp decomp
K0 = torch.rand(S, R, device='cuda')
K1 = torch.rand(H, W, R, device='cuda')
K2 = torch.rand(R, T, device='cuda')


# from paper
def CPForwardPass(U, K0, K1, K2):
	U0 = tnn.conv_einsum("xys, sr -> xyr", U, K0)
	U1 = tnn.conv_einsum("hwr, hwr -> hwr | hw", U0, K1) # here we change xy to hw
	V = tnn.conv_einsum("hwr, rt -> hwt", U1, K2)
	
	return [U0, U1, V]


pass1 = CPForwardPass(U, K0, K1, K2)
print("pass1.size() = " + str(pass1.size()))


dLdV = torch.rand(H, W, T, device='cuda')
#dLdK1 = torch.rand(H, W, R)
#dLdK2 = torch.rand(R, T)
#dLdU0 = torch.rand(X, Y, R)
#dLdU1 = torch.rand(H, W, R)

# from paper
def CPBackwardPass(K1, K2, U, U1, dLdV):
	dLdU1 = tnn.conv_einsum("hwt, tr -> hwr", dLdV, K2.permute(1, 0)) 
	dLdU0 = tnn.conv_einsum("hwr, hwr -> hwr | hw", dLdU1, K1)

	dLdK2 = tnn.conv_einsum("hwr, hwt -> rt", U1, dLdV)	
	dLdK1 = tnn.conv_einsum("hwr, hwr -> hwr | hw", dLdU1.permute(1, 0, 2), U1)
	dLdK0 = tnn.conv_einsum("hws, hwr -> sr", U, dLdU0) 		
 
	return [dLdU0, dLdU1, dLdK0, dLdK1, dLdK2]







