#import sys 
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
#import tnn
from torch_conv_einsumfunc import *

import torch








### einsum example
#
#I = 10
#J = 5
#K = 5
#
#A = torch.ones([I, J])
#B = torch.ones([I, J, K])
#C = torch.ones([K])
#
#T = torch.einsum("ij, ijk, k -> k", A, B, C)
#print(T)
#
## This means T is a tensor defined by the following coordinate formula:
##       for all k, T_k = sum_{i, j} A_ij B_ijk C_k
## Every einsum string translates to a formula in this way. 
## So einsum is a tool for doing arbitrary linear operations with tensors
## For example:
##   - matrix multiplication "ij, jk -> ik"
##   - dot product "i, i ->"
##   - outer product "i, j -> ij"
##   - trace of a square matrix "ii ->"
#
###
#
#
#










### basic convolution T = A * B

I1 = 3
I2 = 5 # note I1 need not equal I2 in a convolution, but the indices much match in size for contraction.

A = torch.ones([I1])
B = torch.ones([I2])

T = conv_einsum("i, i -> i | i", A, B)
print(T)
# So we extend einsum's notation by adding the | section. All indices following
# the | are convolution indices.
#
# You can think about this operation in a parallel way to the interpretation I gave
# above for einsum. Up to different ways of interpreting padding modes and padding amounts,
# the coordinate formula for T is 
#       for all i,  T_i = sum_{i1} A_i1 B_{i - i1}
# A coordinate formula like this applies to all operations computable by conv_einsum.
#
###






#### A more complicated N-ary operation
# (don't run currently broken)
#I = 10
#J = 20
#K1 = 10
#K2 = 10
#L1 = 5
#L2 = 5
#L3 = 5
#
#A = torch.ones([I, J, K1, L1])
#B = torch.ones([I, J, K2, L2])
#C = torch.ones([L3])
#
#T = conv_einsum("ijkl, ijkl, l -> kl | kl", A, B, C)
#print(T.size())
## So you can see that our operation extends einsum by allowing for both contraction indices 
## and for convolution indices.
#
#
## This above example is the same in a single operation as doing two pairwise operations (possibly in the following order)
#T1 = conv_einsum("ijkl, ijkl -> kl | kl", A, B)
#T2 = conv_einsum("kl, l -> | kl", T1, C)
#print(T2.size())
#
####

#### An example of a "full convolution"
#
#I1 = 5
#I2 = 9
#
#A = torch.ones(I1)
#B = torch.ones(I2)
#
#padding = I1 - 1
#
### this is a full convolution since padding = ker_size - 1 and padding_mode='zeros',
### The output size should be o = i + k - 1 
#T = conv_einsum_pair("i, i -> i | i", A, B, \
#                     max_mode_sizes={"i":I1 + I2 - 1}, padding_mode='zeros', padding={"i":I1 - 1})
#print(T)
#
####

#### An example with hyperparameters
#
#I = 10
#J = 5
#
#
#A = torch.ones([I, J])
#B = torch.ones([I, J])
#
#T = conv_einsum("ij, ij -> ij | ij", A, B, \
#                stride={"i":5,"j":1}, dilation={"i":1,"j":1})
#                                           
#print(T.size())
###


### Now I'll explain how conv_einsum works internally. If no convolution indices appear
### then we simply pass all of the inputs along to Pytorch's einsum function. 
### Otherwise assume there is a convolution index.
### There are 3 levels of increasing generality: an atomic level, a pairwise (or 2-way) level, 
### and an N-way level.
###  - The atomic level makes a single call to one of Pytorch's Conv1d/Conv2d/Conv3d functions.
###  - The pairwise/2-way level permutes+reshapes the 2 inputs into a manner suitable for processing by the atomic
###    level, and permutes+reshapes the result from the atomic level back into the expected output.
###  - The N-way level reduces an N-way operation into a sequence of pairwise/2-way operations.
###    It uses a "pairwise sequencer" to find an efficient sequence.

### A major goal we had was to get GPU support for convolutions. Originally we wanted to
### approach this by writing CUDA code directly, but later we figured out that we can do
### this in a better way by reducing conv_einsum internally to Pytorch's Conv1d, Conv2d, 
### or Conv3d functions. We do that by implementing a specific conv_einsum for each case 
### of 1d, 2d, and 3d. 









### For the 1d atomic operation the conv_einsum string we implement is "ijkl, mikl -> mijl | l". 
### This is essentially the index form of a single call to Conv1d with grouping
### (up to minor reshaping which doesn't have to move the data).
### Therefore this operation can be computed with a single call to Conv1d, without permuting indices.
### 
#I = 10
#J = 10
#K = 10
#L1 = 10
#L2 = 20
#M = 10
#
#
#A = torch.ones([I, J, K, L1])
#B = torch.ones([M, I, K, L2])
#
#
#
#padding = max_zeros_padding_1d(L1, L2, 0)
#T = ijkl_mikl_to_mijl_bar_l(A, B, padding_mode='zeros', padding=padding)
#print(T.size())
### let's look inside to see how it works

###
### The 2d and 3d atomic operations are similar









### The pairwise level of the hierarchy is implemented in a function called einsum_pair
### This function calculates all of the permutations / tensor sizes necessary for reshaping to
### and from the atomic operations




### The N-ary level of the hierchy is conv_einsum itself. This biggest component of this
### level is the pairwise sequencer.













