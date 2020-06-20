import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import tnn 

import torch
#
#X = 2 # 128
#Y = 4 # 512
#
#R = 2 # 8
#S = 2 # 3
#T = 2 # 5
#H = 2 # 1024
#W = 2 # 1024
#
#
#U = torch.ones(X, Y, S, device='cpu')
#
#K = torch.ones(H, W, S, T, device='cpu')
## obtained by cp decomp
#K0 = torch.ones(S, R, device='cpu')
#K1 = torch.ones(H, W, R, device='cpu')
#K2 = torch.ones(R, T, device='cpu')
#

# from paper
def CPForwardPass(U, K0, K1, K2):
    U0 = tnn.conv_einsum("xys, sr -> xyr", U, K0)
    U1 = tnn.conv_einsum("hwr, hwr -> hwr | hw", U0, K1) # here we change xy to hw
    V = tnn.conv_einsum("hwr, rt -> hwt", U1, K2)
    
    return [U0, U1, V]


#cp_pass = CPForwardPass(U, K0, K1, K2)



#dLdV = torch.ones(H, W, T, device='cpu')
#dLdK1 = torch.ones(H, W, R)
#dLdK2 = torch.ones(R, T)
#dLdU0 = torch.ones(X, Y, R)
#dLdU1 = torch.ones(H, W, R)

# from paper
def CPBackwardPass(K1, K2, U, U1, dLdV):
    dLdU1 = tnn.conv_einsum("hwt, tr -> hwr", dLdV, K2.permute(1, 0)) 
    dLdU0 = tnn.conv_einsum("hwr, hwr -> hwr | hw", dLdU1, K1)

    dLdK2 = tnn.conv_einsum("hwr, hwt -> rt", U1, dLdV)    
    dLdK1 = tnn.conv_einsum("hwr, hwr -> hwr | hw", dLdU1.permute(1, 0, 2), U1)
    dLdK0 = tnn.conv_einsum("hws, hwr -> sr", U, dLdU0)         
 
    return [dLdU0, dLdU1, dLdK0, dLdK1, dLdK2]



#dLdV = torch.ones(H, W, T, device='cpu')
#dLdK1 = torch.ones(H, W, R)
#dLdK2 = torch.ones(R, T)
#dLdU0 = torch.ones(X, Y, R)
#dLdU1 = torch.ones(H, W, R)

# from paper
def CPBackwardPass(K1, K2, U, U1, dLdV):
    dLdU1 = tnn.conv_einsum("hwt, tr -> hwr", dLdV, K2.permute(1, 0)) 
    dLdU0 = tnn.conv_einsum("hwr, hwr -> hwr | hw", dLdU1, K1)

    dLdK2 = tnn.conv_einsum("hwr, hwt -> rt", U1, dLdV)    
    dLdK1 = tnn.conv_einsum("hwr, hwr -> hwr | hw", dLdU1.permute(1, 0, 2), U1)
    dLdK0 = tnn.conv_einsum("hws, hwr -> sr", U, dLdU0)         
 
    return [dLdU0, dLdU1, dLdK0, dLdK1, dLdK2]


#######################################################

# from paper
def TKForwardPass(U, K0, K1, K2):
    U0 = tnn.conv_einsum("xys, sr -> xyr", U, K0)
    U1 = tnn.conv_einsum("hwr, hwrl -> hwl | hw", U0, K1) # here we change xy to hw
    V = tnn.conv_einsum("hwl, lt -> hwt", U1, K2)
    
    return [U0, U1, V]

def TKForwardPass2(U, K0, K1, K2):
    U0 = tnn.conv_einsum("xys, sr -> xyr", U, K0)        
    V = tnn.conv_einsum("hwr, hwrl, lt -> hwt | hw", U0, K1, K2)

    return [U0, V]

def TKForwardPass3(U, K0, K1, K2):
    V = tnn.conv_einsum("hws, sr, hwrl, lt -> hwt | hw", U, K0, K1, K2)
    return V




X = 2 # 128
Y = 4 # 512

R = 2 # 8
L = 2 # -
S = 2 # 3
T = 2 # 5
H = 2 # 1024
W = 2 # 1024


U = torch.ones(X, Y, S, device='cpu')

K = torch.ones(H, W, S, T, device='cpu')
# obtained by cp decomp
K0 = torch.ones(S, R, device='cpu')
K1 = torch.ones(H, W, R, L, device='cpu')
K2 = torch.ones(R, T, device='cpu')


tk_pass = TKForwardPass(U, K0, K1, K2)
print("tk_pass U0 = \n" + str(tk_pass[0]))
print("tk_pass V = \n" + str(tk_pass[2]))


tk_pass2 = TKForwardPass2(U, K0, K1, K2)

print("tk_pass2 U0 = \n" + str(tk_pass2[0]))
print("tk_pass2 V = \n" + str(tk_pass2[1]))



tk_pass3 = TKForwardPass3(U, K0, K1, K2)
print("tk_pass3 V = \n" + str(tk_pass3))


dLdV = torch.ones(H, W, T, device='cpu')







#def TestPass1(A, B, C):
#    D = tnn.conv_einsum("ij, jl -> il | j", A, B)
#    print("D.size(): " + str(D.size()) + "\n")
#    E = tnn.conv_einsum("il, l -> i", D, C)
#
#    return [D, E]
#
#def TestPass2(A, B, C):
#    E = tnn.conv_einsum("ij, jl, l -> i | j", A, B, C)
#
#    return E
#
#I = 2
#J = 3
#L = 4
#A = torch.ones(I, J)
#B = torch.ones(J, L)
#C = torch.ones(L)
#
#test_pass1 = TestPass1(A, B, C)
#print("test_pass1 E = \n" + str(test_pass1[1]))
#
#test_pass2 = TestPass2(A, B, C)
#print("test_pass2 E = \n" + str(test_pass2))


#def TestPass1(A, B, C):
#    D = tnn.conv_einsum("i, l -> il", A, B) 
#    E = tnn.conv_einsum("il, l -> i", D, C)
#
#    #D = torch.einsum("i, l -> il", A, B) 
#    #E = torch.einsum("il, l -> i", D, C)
#
#
#    return [D, E]
#
#def TestPass2(A, B, C):
#    E = tnn.conv_einsum("i, l, l -> i", A, B, C)
#    #E = torch.einsum("i, l, l -> i", A, B, C)
#
#    return E
#
#I = 2
#L = 4
#A = torch.ones(I)
#B = torch.ones(L)
#C = torch.ones(L)
#
#test_pass1 = TestPass1(A, B, C)
#print("test_pass1 E = \n" + str(test_pass1[1]) + "\n\n\n\n\n\n\n\n\n\n\n\n")
#
#test_pass2 = TestPass2(A, B, C)
#print("test_pass2 E = \n" + str(test_pass2))



#I = 2
#J = 2
#K = 2
#L = 2
#
#A = torch.ones(I, J, K)
#B = torch.ones(I, K, J)
#C = torch.ones(J, K, I)
#D = torch.ones(I)
#E = torch.ones(K)
#F = torch.ones(L, K)
#
#T1 = torch.einsum("ijk, ikj, jki, i, k, lk -> jki", A, B, C, D, E, F)
#print("torch_einsum = \n" + str(T1))
#
#T2 = tnn.conv_einsum("ijk, ikj, jki, i, k, lk -> jki", A, B, C, D, E, F)
#print("tnn_conv_einsum = \n" + str(T2))
#
#
