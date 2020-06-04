
import torch

from parse_conv_einsum import _parse_conv_einsum_input
#from parse_conv_einsum import *




##torch.nn.Conv1d
#batch_size = 1
#signal_length = 8
#kernel_size = 3
#in_channels = 1
#out_channels = 1
#padding = 1
#
#m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
#m.weight.data = torch.tensor([[[1., 2., 3.]]])
#print("weight" + str(m.weight.data))
#print("\n")
##A = torch.randn(batch_size, in_channels, signal_length)
#A = torch.tensor([[[1., 2., 3., 4., -20., 300., -21., 22.]]])
#print(A)
#output = m(A)
#print(output)
#
#


def padding_1d(ker_size, input_size):     
    # see https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html for the definition
    # under the Shape section for the definition of the padding. We add 1 so the integer division
    # by 2 errs large instead of small
    # for 1d convolutions ker_size < input_size but this is not true for higher dimensions, and I
    # use this function to compute those paddings
    return (max(ker_size, input_size) - input_size + ker_size)//2

def padding_2d(ker_size, input_size):
    return (padding_1d(ker_size[0], input_size[0]), padding_1d(ker_size[1], input_size[1]))


def padding_3d(ker_size, input_size):
    return (padding_1d(ker_size[0], input_size[0]), padding_1d(ker_size[1], input_size[1]), padding_1d(ker_size[2], input_size[2]))

def padding_nd(ker_size, input_size):
    return tuple(padding_1d(ker_size[i], input_size[i]) for i in range(0,len(ker_size)))


def convolve_same(vec1, vec2):
    # this function should agree with the convolve functions offered by numpy and scipy, with mode='same' 
    kernel_size = len(vec1)
    input_size = len(vec2)
 
    if(kernel_size > input_size): # Conv1d requires the kernel to be shorter than the input
        return convolve_same(vec2, vec1)
  
    groups = 1
    in_channels = 1
    out_channels = 1
    
    
    padding = padding_1d(kernel_size, input_size)   
    
    # \todo I'm not sure on which inputs the output vector will have the right length so I'm just defensively truncating it
    
    
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    m.weight.data[0, 0, :] = torch.flip(vec1, [0])

    output = m(vec2.reshape(1, 1, input_size))
    return output.data[0,0,:input_size]


#
#A = [ 5., 5., 5.]
#B = [1., 2.]
#
#torch_A = torch.tensor(A)
#torch_B = torch.tensor(B)
#print("Torch: " + str(convolve_same(torch_A, torch_B)))
#
#import numpy as np
#print("Numpy: " + str(np.convolve(A, B, mode='same')))
#



def independent_convolve_same(mat1, mat2):
    # the point of this function is to try to compute many convolutions using one call to Conv1d
    # in einsum notation this computes "ij, ij -> ij | j"

    num_convolutions = mat1.size(0) 
    if(num_convolutions != mat2.size(0)):
        print("Error: The inputs must have the same number of rows")

    kernel_size = mat1.size(1)
    input_size = mat2.size(1)
 
    if(kernel_size > input_size): # Conv1d requires the kernel to be shorter than the input
        return convolve_same(mat2, mat1)
 
    groups = num_convolutions
    in_channels = groups 
    out_channels = groups 
    

    # this padding is chosen so that input_size + 2*padding - kernel_size + 1 >= input_size, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    padding = padding_1d(kernel_size, input_size) 
  
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    mat1_flipped = torch.flip(mat1, [1])
    m.weight.data = mat1_flipped.view(out_channels, in_channels//groups, kernel_size)

    output = m(mat2.reshape(1, in_channels, input_size))

    return torch.squeeze(output.data[:,:,:input_size], 0)
    


#A = [ [1., 2.], [3., 4.]]
#B = [ [2., -3.], [14., 24.]]
#
#torch_A = torch.tensor(A)
#torch_B = torch.tensor(B)
#
#print(torch_A)
#print(torch_B)
#print("Torch: " + str(independent_convolve_same(torch_A, torch_B)))
#
#print("single1: " + str(torch_A[1, :]))
#print("single2: " + str(torch_B[1, :]))
#print("single: " + str(convolve_same(torch_A[1, :], torch_B[1, :])))
#import numpy as np
#print("numpy: " + str(np.convolve(list(torch_A[1, :]), list(torch_B[1, :]), mode='same')))

def ij_ij_to_ij_bar_j(tens1, tens2):
    return independent_convolve_same(tens1, tens2)

#torch_A = torch.tensor([ [1., 2.], [3., 4.]])
#torch_B = torch.tensor([ [2., -3.], [14., 24.]])
#print("'ij,ij->ij|j' = \n" + str(ij_ij_to_ij_bar_j(torch_A, torch_B)))




def ij_ij_to_j_bar_j(tens1, tens2):    
    # this is a simpler version of the atomic operation we want. We could compute this
    # using ij_ij_to_ij_bar_j plus "ij->j" but we'd like to prevent having a larger intermediate object
    # this is also so I can test the capabilities of Conv1d

    num_convolutions = tens1.size(0) 

    kernel_size = tens1.size(1)
    input_size = tens2.size(1)
 
    if(kernel_size > input_size): # Conv1d requires the kernel to be shorter than the input
        return convolve_same(tens2, tens1)
 
    groups = 1
    in_channels = num_convolutions
    out_channels = 1
    

    # this padding is chosen so that input_size + 2*padding - kernel_size + 1 >= input_size, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    padding = padding_1d(kernel_size, input_size) 
  
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [1])
    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size)

    output = m(tens2.reshape(1, in_channels, input_size))

    return torch.squeeze(torch.squeeze(output.data[:,:,:input_size], 0), 0)
   


#torch_A = torch.tensor([ [1., 2.], [3., 4.]])
#torch_B = torch.tensor([ [2., -3., 5], [14., 24., 5]])
#print("'ij,ij->ij|j' = \n" + str(ij_ij_to_ij_bar_j(torch_A, torch_B)))
#print("'ij->j' = \n" + str(torch.einsum("ij->j", ij_ij_to_ij_bar_j(torch_A, torch_B))))
#print("ij_ij_to_j_bar_j = \n" + str(ij_ij_to_j_bar_j(torch_A, torch_B)))



def ijk_ijk_to_ik_bar_k(tens1, tens2):    
    # this is a simpler version of the atomic operation we want. We could compute this
    # using ij_ij_to_ij_bar_j plus "ij->j" but we'd like to prevent having a larger intermediate object
    # this is also so I can test the capabilities of Conv1d

    kernel_size = tens1.size(2)
    input_size = tens2.size(2)
 
    if(kernel_size > input_size): # Conv1d requires the kernel to be shorter than the input
        return convolve_same(tens2, tens1)
 
    groups = tens1.size(0)
    in_channels = groups*tens1.size(1)
    out_channels = groups
    

    # this padding is chosen so that input_size + 2*padding - kernel_size + 1 >= input_size, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    padding = padding_1d(kernel_size, input_size) 
  
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [2])
    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size)

    output = m(tens2.reshape(1, in_channels, input_size))

    return torch.squeeze(torch.squeeze(output.data[:,:,:input_size], 0), 0)
   

##torch_A = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
#torch_A = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]], [[9,10],[11,12]]])
#torch_B = torch_A
##torch_B = torch.tensor([[[11,21],[31,41]],[[51,61],[71,81]]])
#print("ijk_ijk_to_ik_bar_k = \n" + str(ijk_ijk_to_ik_bar_k(torch_A, torch_B)))
#print("0jk_0jk_to_0k_bar_k = \n" + str(ij_ij_to_j_bar_j(torch_A[0,:], torch_A[0,:])))
#print("1jk_1jk_to_1k_bar_k = \n" + str(ij_ij_to_j_bar_j(torch_B[1,:], torch_B[1,:])))
#print("2jk_2jk_to_2k_bar_k = \n" + str(ij_ij_to_j_bar_j(torch_B[2,:], torch_B[2,:])))



def ij_j_to_ij_bar_j(tens1, tens2):
    # this is a simpler version of the atomic operation we want. We could compute this
    # using ij_ij_to_ij_bar_j plus "ij->j" but we'd like to prevent having a larger intermediate object
    # this is also so I can test the capabilities of Conv1d

    kernel_size = tens1.size(1)
    input_size = tens2.size(0)
    conv_len = max(kernel_size, input_size)
 
    groups = 1
    in_channels = 1 
    out_channels = tens1.size(0)
    

    # this padding is chosen so that input_size + 2*padding - kernel_size + 1 >= input_size, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    padding = padding_1d(kernel_size, input_size) 
  
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [1])
    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size)
    

    output = m(tens2.reshape(1, in_channels, input_size))
    
    return torch.squeeze(output.data[:,:,:conv_len], 0)
   

#torch_A = torch.tensor([[1.,2.], [4.,5.]])
#torch_B = torch.tensor([1.,2.,3.,4.,19.,29.])
#
##torch_A = torch.tensor([[1.,2.,3.,4.,-5.], [4.,5.,6.,7.,19.]])
##torch_B = torch.tensor([1.,2.,3.])
#
#
#print("torch_A =\n" + str(torch_A))
#print("torch_B =\n" + str(torch_B))
#print("'ij,j->ij|j' = \n" + str(ij_j_to_ij_bar_j(torch_A, torch_B)))
#print("A0: * B =\n" + str(convolve_same(torch_A[0,:], torch_B)))


def ijk_ik_to_ijk_bar_k(tens1, tens2):
    kernel_size = tens1.size(2)
    input_size = tens2.size(1)
    conv_len = max(kernel_size, input_size)
 
    groups = tens1.size(0)
    in_channels = groups
    out_channels = groups*tens1.size(1)
    

    # this padding is chosen so that input_size + 2*padding - kernel_size + 1 >= input_size, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    padding = padding_1d(kernel_size, input_size) 
  
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [2])
    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size)
    

    output = m(tens2.reshape(1, in_channels, input_size))
    
    return torch.reshape(torch.squeeze(output.data[:,:,:conv_len], 0), (tens1.size(0), tens1.size(1), conv_len))
    
    

#torch_A = torch.tensor([[[1.,2.], [3.,4.], [5.,6.]], [[7.,8.], [9.,10.], [11.,12.]]])
#torch_B = torch.tensor([[1.,2.], [3.,4.]])
#
#print("torch_A =\n" + str(torch_A))
#print("torch_B =\n" + str(torch_B))
#print("test = \n" + str(ijk_ik_to_ijk_bar_k(torch_A, torch_B)))
#print("0jk_0k_to_0jk_bar_k = \n" + str(ij_j_to_ij_bar_j(torch_A[0,:,:], torch_B[0,:])))



def ijkl_ikl_to_ijl_bar_l(tens1, tens2):
    # this is supposed to compute the most general, up to reshaping / permuting, convolutional einsum 
    # with 1 convolution index which is computable by Conv1d alone

    kernel_size = tens1.size(3)
    input_size = tens2.size(2)
    conv_len = max(kernel_size, input_size)
    
 
    groups = tens1.size(0)
    in_channels = groups*tens1.size(2)
    out_channels = groups*tens1.size(1)
    

    # this padding is chosen so that input_size + 2*padding - kernel_size + 1 >= input_size, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    padding = padding_1d(kernel_size, input_size) 
  
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [3])
    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size)
    

    output = m(tens2.reshape(1, in_channels, input_size))
    
    return torch.reshape(torch.squeeze(output.data[:,:,:conv_len], 0), (tens1.size(0), tens1.size(1), conv_len))



#A = [[[[1.,2.], [3.,4.], [5.,6.]], [[7.,8.], [9.,10.], [11.,12.]]],
#     [[[1.,2.], [3.,4.], [5.,6.]], [[7.,8.], [9.,10.], [11.,12.]]],
#     [[[1.,2.], [3.,4.], [5.,6.]], [[7.,8.], [9.,10.], [11.,12.]]],
#     [[[1.,2.], [3.,4.], [5.,6.]], [[7.,8.], [9.,10.], [11.,12.]]]]
#torch_A = torch.tensor(A)
#
#B = [[[1.], [3.], [5.]],
#     [[1.], [3.], [5.]],
#     [[1.], [3.], [5.]],
#     [[1.], [3.], [5.]]]
#
##B = [[[1.,2.,3.], [3., 4., 3.], [5., 6, 3.]],
##     [[1.,2.,3.], [3., 4., 3.], [5., 6, 3.]],
##     [[1.,2.,3.], [3., 4., 3.], [5., 6, 3.]],
##     [[1.,2.,3.], [3., 4., 3.], [5., 6,3.]]]
#
#torch_B = torch.tensor(B)
#
#print("torch_A = \n" + str(torch_A))
#print("torch_A.size() = \n" + str(torch_A.size()))
#print("torch_B = \n" + str(torch_B))
#print("torch_B.size() = \n" + str(torch_B.size()))
#
#print("'ijkl, ikl -> ijl | l' = \n" + str(ijkl_ikl_to_ijl_bar_l(torch_A, torch_B)))
#print("'ij0l, i0l -> ijl | l' = \n" + str(ijk_ik_to_ijk_bar_k(torch_A[:,:,0,:], torch_B[:,0,:])))
#print("'ij1l, i1l -> ijl | l' = \n" + str(ijk_ik_to_ijk_bar_k(torch_A[:,:,1,:], torch_B[:,1,:])))
#print("'ij2l, i2l -> ijl | l' = \n" + str(ijk_ik_to_ijk_bar_k(torch_A[:,:,2,:], torch_B[:,2,:])))
#




# testing reshaping output indices
# "ijk,k->ij" computable with "ik,k->i"?
#torch_A = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
#torch_B = torch.tensor([1,2])
#reshaped_A = torch.reshape(torch_A, (4,2))
#print("einsum('ijk,k->ij') =\n" + str(torch.einsum("ijk,k->ij", torch_A, torch_B)))
#print("einsum('ik,k->ij') =\n" + str(torch.reshape(torch.einsum("ik,k->i", reshaped_A, torch_B), (2,2))))



## is "ij,kl->ikjl" computable with "i,j->ij"?
#torch_A = torch.tensor([[1,2],[3,4]])
#torch_B = torch.tensor([[5,6],[7,8]])
#print("einsum('ij,kl->ikjl') = \n" + str(torch.einsum("ij,kl->ikjl",torch_A,torch_B)))
#reshaped_A = torch.reshape(torch_A,(4,))
#reshaped_B = torch.reshape(torch_B,(4,))
#eins = torch.einsum("i,j->ij", reshaped_A, reshaped_B)
#reshaped_eins = torch.reshape(eins, (2,2,2,2))
#permuted_eins = reshaped_eins.permute(0,2,1,3)
###permuted_eins = torch.einsum("ijkl->ikjl", reshaped_eins)
#print("einsum('i,j->ij') = \n" + str(permuted_eins))



# \todo Need to test what the fastest arrangement of indices is
def ijkl_mikl_to_mijl_bar_l_forloop(tens1, tens2):
    # we need this additional atomic einsum, but it is not computable with Conv1d alone 
    # that is what I thought until Jiahao explained how to include batch_size; so I implemented this without a forloop
    out = torch.zeros(tens2.size(0), tens1.size(0), tens1.size(1), max(tens1.size(3), tens2.size(3)))
   
    for m in range(0, tens2.size(0)):
        out[m,:,:,:] = ijkl_ikl_to_ijl_bar_l(tens1, tens2[m,:,:,:]) 
    
    return out    


#A = [[[[1.,2.], [3.,4.], [5.,6.]], [[7.,8.], [9.,10.], [11.,12.]]],
#     [[[11.,21.], [31.,41.], [51.,61.]], [[71.,81.], [91.,101.], [111.,121.]]],
#     [[[1.,2.], [3.,4.], [5.,6.]], [[7.,8.], [9.,10.], [11.,12.]]],
#     [[[1.,2.], [3.,4.], [5.,6.]], [[7.,8.], [9.,10.], [11.,12.]]]]
#torch_A = torch.tensor(A)
#torch_B = torch.ones(torch_A.size(0), 3, torch_A.size(2), torch_A.size(3))
#print("'ijkl, imkl -> ijml | l' = \n" + str(ijkl_imkl_to_imjl_bar_l_forloop(torch_A, torch_B)))
#print("'ijkl, i0kl -> ij0l | l' = \n" + str(ijkl_ikl_to_ijl_bar_l(torch_A, torch_B[:,0,:,:].reshape(torch_A.size(0), torch_A.size(2), torch_A.size(3)))))
#print("'ijkl, i1kl -> ij1l | l' = \n" + str(ijkl_ikl_to_ijl_bar_l(torch_A, torch_B[:,1,:,:].reshape(torch_A.size(0), torch_A.size(2), torch_A.size(3)))))



def ijkl_mikl_to_mijl_bar_l(kernel, input_tens):
    # this is supposed to compute the most general, up to reshaping / permuting, convolutional einsum 
    # with 1 convolution index which is computable by Conv1d alone
    # This is the order the indices must appear in so that the operation can be done without any calls to permute

    # by the Conv1d documentation https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
    # input = (batch_size = M, input_channel = groups * K = I * K, input_size = input_tens.size(3))
    # weight = (out_channels = groups * J = I * J, in_channels/groups = K, kernel_size = kernel.size(3))
    # output = (batch_size = M, out_channels = groups * J = I * J, conv_len = max(input_size, kernel_size))
    # and it appears the indices are laid out as (so, after reshaping)
    # input: M, I, K, L
    # weight: I, J, K, L
    # output: M, I, J, L

    batch_size = input_tens.size(0)
    kernel_size = kernel.size(3)
    input_size = input_tens.size(3)
    conv_len = max(kernel_size, input_size)
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)

    # this padding is chosen so that input_size + 2*padding - kernel_size + 1 >= input_size, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    padding = padding_1d(kernel_size, input_size) 
  
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    kernel_flipped = torch.flip(kernel, [3])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size)
    
    output = m(input_tens.reshape(batch_size, in_channels, input_size))  
     
    return torch.reshape(output.data[:,:,:conv_len], (batch_size, groups, out_channels//groups, conv_len))
    
    



#torch_A = torch.randn(4,5,6,3)
#torch_B = torch.randn(3, torch_A.size(0), torch_A.size(2), 5)
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#eins = ijkl_mikl_to_mijl_bar_l(torch_A, torch_B)
#eins_forloop = ijkl_mikl_to_mijl_bar_l_forloop(torch_A, torch_B)
#print("'ijkl, imkl -> imjl | l' = \n" + str(eins.size()))
#print("'ijkl, imkl -> imjl | l' = \n" + str(eins))
#print("'ijkl, imkl -> imjl | l' = \n" + str(eins_forloop))
#print("eins - eins_forloop = \n" + str(eins - eins_forloop))
##print("'ijkl, i0kl -> ij0l | l' = \n" + str(ijkl_ikl_to_ijl_bar_l(torch_A, torch_B[:,0,:,:].reshape(torch_A.size(0), torch_A.size(2), torch_A.size(3)))))
##print("'ijkl, i1kl -> ij1l | l' = \n" + str(ijkl_ikl_to_ijl_bar_l(torch_A, torch_B[:,1,:,:].reshape(torch_A.size(0), torch_A.size(2), torch_A.size(3)))))


def convolve2d_same(mat1, mat2):
    
    num_convolutions = 1  

    kernel_size = mat1.size()
    input_size = mat2.size() 
 
    max_h = max(kernel_size[0], input_size[0])
    max_w = max(kernel_size[1], input_size[1]) 
 
    in_channels = num_convolutions
    out_channels = num_convolutions
    groups = num_convolutions

   
    padding = padding_2d(kernel_size, input_size)
    
    m = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    mat1_flipped = torch.flip(mat1, [0,1])

    m.weight.data = mat1_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])

    output = m(mat2.view(1, in_channels, input_size[0], input_size[1]))

    # output.data is a 1 x 1 x width x height tensor
    # and we slice it to the proper dimensions, because Conv2d returns too many
    # rows and columns with the best possible padding
    return torch.squeeze(torch.squeeze(output.data[:,:,:max_h,:max_w], 0),0) 
    

    



#A = [[1.,1.,1.], [1.,1.,1.], [1.,1.,1.]]
#B = [[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]
#
#torch_A = torch.tensor(A)
#torch_B = torch.tensor(B)
#print("torch_A = \n" + str(torch_A))
#print("torch_B = \n" + str(torch_B))
#print("2d convolve = \n" + str(convolve2d_same(torch_A, torch_B)))
#
#import numpy as np
#from scipy import signal
#print("signal conv = \n" + str(signal.convolve2d(np.array(A), np.array(B), mode='full')))
#




#
##B = [[1.,1.,1.], [1.,1.,1.], [1., 1.,1.],[1., 1.,1.]]
#B = [[1.,1.,1.,1.]]
##A = [[1.,1.], [1.,1.]]
##A = [[1.]]
#A = [[1.],[1.],[1.],[1.]]
##B = [[1.,1.,1.,1.], [1.,1.,1.,1.], [1.,1.,1.,1.], [1.,1.,1.,1.]]
##A = [[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]
#
#torch_A = torch.tensor(A)
#torch_B = torch.tensor(B)
#print("torch_A = \n" + str(torch_A))
#print("torch_B = \n" + str(torch_B) + "\n")
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#
#print("2d convolve = \n" + str(convolve2d_same(torch_A, torch_B)) + "\n")
#
#import numpy as np
#from scipy import signal
#print("signal conv = \n" + str(signal.convolve2d(np.array(A), np.array(B), mode='full')) + "\n")
#




#print("torch_A[0] = " + str(torch_A[0]))
#print("torch_B[1] = " + str(torch_B[1]))
#print("conv = " + str(convolve_same(torch_A[0], torch_B[1])))





def independent_convolve2d_same(tens1, tens2):
    
    num_convolutions = tens1.size(0)
    if(num_convolutions != tens2.size(0)):        
        print("Error: the inputs must have the same length in the 0th mode")

    kernel_size = tens1.size()[1:3]
    input_size = tens2.size()[1:3]
 
    max_h = max(kernel_size[0], input_size[0])
    max_w = max(kernel_size[1], input_size[1])
    
 
    in_channels = num_convolutions
    out_channels = num_convolutions
    groups = num_convolutions

   
    padding = padding_2d(kernel_size, input_size)
    
    m = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [1,2])

    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])

    output = m(tens2.reshape(1, in_channels, input_size[0], input_size[1]))

    # output.data is a 1 x 1 x width x height tensor
    # and we slice it to the proper dimensions, because Conv2d returns too many
    # rows and columns, even with the best possible padding
    return torch.squeeze(output.data[:,:,:max_h,:max_w], 0) 




#B = [[[1., 2., 3.]], [[1.,1.,1.]]]
#A = [[[2., -3.], [14., 24.]], [[1.,1.], [1., 1.]]]
#
#torch_A = torch.tensor(A)
#torch_B = torch.tensor(B)
#
#
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#print("torch_A = \n" + str(torch_A))
#print("torch_B = \n" + str(torch_B))
#print("independent_convolve2d_same = \n" + str(independent_convolve2d_same(torch_A, torch_B)) + "\n")
#
#print("single1: " + str(torch_A[0,:,:]))
#print("single2: " + str(torch_B[0,:,:]))
#print("single = \n" + str(convolve2d_same(torch_A[0,:,:], torch_B[0,:,:])) + "\n")



def ijklm_iklm_to_ijlm_bar_lm(tens1, tens2): 

    kernel_size = tens1.size()[3:5]
    input_size = tens2.size()[2:4]
    max_h = max(kernel_size[0], input_size[0])
    max_w = max(kernel_size[1], input_size[1]) 
    groups = tens1.size(0)
    in_channels = groups*tens1.size(2)
    out_channels = groups*tens1.size(1)  
    padding = padding_2d(kernel_size, input_size) 
  
    m = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [3,4])
    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])
    
    output = m(tens2.reshape(1, in_channels, input_size[0], input_size[1]))
    
    return torch.reshape(torch.squeeze(output.data[:,:,:max_h,:max_w], 0), (tens1.size(0), tens1.size(1), max_h, max_w))

#
#torch_A = torch.ones(5,4,2,3,2)
#torch_B = torch.ones(5,2,3,2)
#print("ijklm_iklm_to_ijlm_bar_lm = \n" + str(ijklm_iklm_to_ijlm_bar_lm(torch_A, torch_B)))
#print("000lm_00lm_to_00lm_bar_lm = \n" + str(convolve2d_same(torch_A[0,0,0,:,:], torch_B[0,0,:,:])))
#print("001lm_01lm_to_00lm_bar_lm = \n" + str(convolve2d_same(torch_A[0,0,1,:,:], torch_B[0,1,:,:])))



def ijklm_niklm_to_nijlm_bar_lm_forloop(tens1, tens2):
    # we need this additional atomic einsum, but it is not computable with Conv2d alone 
    # that is what I thought until Jiahao explained how to include batch_size; so I implemented this without a forloop
    out = torch.zeros(tens2.size(0), tens1.size(0), tens1.size(1), max(tens1.size(3), tens2.size(3)), max(tens1.size(4), tens2.size(4)))
   
    for n in range(0, tens2.size(0)):
        out[n,:,:,:,:] = ijklm_iklm_to_ijlm_bar_lm(tens1, tens2[n,:,:,:,:]) 
    
    return out

def ijklm_niklm_to_nijlm_bar_lm(kernel, input_tens):
    # this is supposed to compute the most general, up to reshaping / permuting, convolutional einsum 
    # with 2 convolution indices which is computable by Conv2d alone

    batch_size = input_tens.size(0)
    kernel_size = kernel.size()[3:5]
    input_size = input_tens.size()[3:5]
    max_h = max(kernel_size[0], input_size[0])
    max_w = max(kernel_size[1], input_size[1]) 
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)
    padding = padding_2d(kernel_size, input_size) 
  
    m = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    kernel_flipped = torch.flip(kernel, [3,4])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])
    
    output = m(input_tens.reshape(batch_size, in_channels, input_size[0], input_size[1]))
    
    return torch.reshape(output.data[:,:,:max_h,:max_w], (batch_size, groups, out_channels//groups, max_h, max_w))


#torch_A = torch.randn(5,4,2,3,18)
#torch_B = torch.randn(6,5,2,3,12)
#eins = ijklm_niklm_to_nijlm_bar_lm(torch_A, torch_B)
#eins_forloop = ijklm_niklm_to_nijlm_bar_lm_forloop(torch_A, torch_B)
##print("ijklm_niklm_to_nijlm_bar_lm_forloop = \n" + str(eins_forloop))
##print("ijklm_niklm_to_nijlm_bar_lm = \n" + str(eins))
#print("eins - eins_forloop = \n" + str(eins - eins_forloop))


# \todo I think this is correct but I am less certain because numpy/scipy don't provide a 3d convolution for me to test this against
def convolve3d_same(tens1, tens2):
    
    num_convolutions = 1  

    kernel_size = tens1.size()
    input_size = tens2.size() 
 
    max_d = max(kernel_size[0], input_size[0])
    max_h = max(kernel_size[1], input_size[1])
    max_w = max(kernel_size[2], input_size[2]) 
 
    in_channels = num_convolutions
    out_channels = num_convolutions
    groups = num_convolutions

   
    padding = padding_3d(kernel_size, input_size)
    
    m = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [0,1,2])

    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2])

    output = m(tens2.view(1, in_channels, input_size[0], input_size[1], input_size[2]))

    # output.data is a 1 x 1 x depth x width x height tensor
    # and we slice it to the proper dimensions, because Conv3d returns too many
    # rows and columns, even with the best possible padding
    return torch.squeeze(torch.squeeze(output.data[:,:,:max_d,:max_h,:max_w], 0),0) 
 


##A = [[[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]]]
#A = [[[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]]]
#B = [[[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]]]
#
#torch_A = torch.tensor(A)
#torch_B = torch.tensor(B)
#print("torch_A = \n" + str(torch_A))
#print("torch_B = \n" + str(torch_B) + "\n")
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#
#print("3d convolve = \n" + str(convolve3d_same(torch_A, torch_B)) + "\n")



# \todo I think this is correct but I am less certain because numpy/scipy don't provide a 3d convolution for me to test this against
def independent_convolve3d_same(tens1, tens2):
    
    num_convolutions = tens1.size(0)
    if(num_convolutions != tens2.size(0)):        
        print("Error: the inputs must have the same length in the 0th mode")

    kernel_size = tens1.size()[1:4]
    input_size = tens2.size()[1:4]
  
    max_d = max(kernel_size[0], input_size[0])
    max_h = max(kernel_size[1], input_size[1])
    max_w = max(kernel_size[2], input_size[2]) 
    
 
    in_channels = num_convolutions
    out_channels = num_convolutions
    groups = num_convolutions

   
    padding = padding_3d(kernel_size, input_size)
    
    m = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    tens1_flipped = torch.flip(tens1, [1,2,3])

    m.weight.data = tens1_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2])

    output = m(tens2.view(1, in_channels, input_size[0], input_size[1], input_size[2]))

    # output.data is a 1 x 1 x width x height tensor
    # and we slice it to the proper dimensions, because Conv2d returns too many
    # rows and columns, even with the best possible padding
    return torch.squeeze(output.data[:,:,:max_d,:max_h,:max_w], 0)

##A = [[[[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]]]]
#A = [[[[1.], [1.]], [[1.], [1.]], [[1.], [1.]], [[1.], [1.]]]]
#B = [[[[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]], [[1.,1.], [1.,1.]]]]
#
#torch_A = torch.tensor(A)
#torch_B = torch.tensor(B)
#print("torch_A = \n" + str(torch_A))
#print("torch_B = \n" + str(torch_B) + "\n")
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#
#print("3d convolve = \n" + str(independent_convolve3d_same(torch_A, torch_B)) + "\n")





def ijklmn_oiklmn_to_oijlmn_bar_lmn(kernel, input_tens):
    # i.e "ijklmn, oiklmn -> oijlmn | lmn"
    # This is supposed to compute the most general, up to reshaping / permuting, convolutional einsum 
    # with 3 convolution indices, which is computable by Conv3d alone.
    # The index order is what's produced by Conv3d, without permuting.

    batch_size = input_tens.size(0)
    kernel_size = kernel.size()[3:6]
    input_size = input_tens.size()[3:6]
    max_d = max(kernel_size[0], input_size[0])
    max_h = max(kernel_size[1], input_size[1])
    max_w = max(kernel_size[2], input_size[2]) 
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)  
    padding = padding_3d(kernel_size, input_size) 
  
    m = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    kernel_flipped = torch.flip(kernel, [3,4,5])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2])
    
    output = m(input_tens.reshape(batch_size, in_channels, input_size[0], input_size[1], input_size[2]))
    
    return torch.reshape(output.data[:,:,:max_d,:max_h,:max_w], (batch_size, groups, out_channels//groups, max_d, max_h, max_w))

torch_A = torch.randn(100,4,2,3,180,1)
torch_B = torch.randn(6,100,2,30,120,1)

#torch_A = torch.randn(2,2,2,2,2,1)
#torch_B = torch.randn(2,2,2,2,2,1)

eins = ijklmn_oiklmn_to_oijlmn_bar_lmn(torch_A, torch_B)
eins2d = ijklm_niklm_to_nijlm_bar_lm(torch_A[:,:,:,:,:,0], torch_B[:,:,:,:,:,0])
print((torch.squeeze(eins) - eins2d).sum())




def occurrence_indices(collection, possible_occurrences):
    return [i for i in range(0, len(collection)) if collection[i] in possible_occurrences]

def permuted_collection(collection, permutation):
    # returns list
    permuted = [0]*len(permutation) 
    for i in range(0, len(permutation)):
        permuted[permutation[i]] = collection[i]

    return permuted


# this function computes the special case of conv_einsums where the only operations are N-dimensional convolution
# and elementwise multiplication
def conv_einsum_pair_convolution_only(*operands):
    # \type This function assumes the einsum string is of a particular type
    input_subscripts, output_subscript, convolution_subscript, subscripts_set, operands \
        = _parse_conv_einsum_input(operands) 

    # \todo just assuming there's only a single convolution for now, will add support for N dimensional convolutions
    # actually, I'll try to implement 2 and 3 way convolutions first, because that will give me a better idea for what
    # to expect when implementing this method
    
    print("input_subscripts " + str(input_subscripts))
    
    # \todo I slapped together the code to compute these permutations of the two input_subscripts such that
    # the convolutions would occur at the end, and the second permuted subscript would have its convolutions
    # appear in the same order as the first permuted subscript. I think there are better ways of writing
    # this code. Aside from code cleanliness improvements, of which many are possible (including better naming), 
    # it might be better if the permutations minimize the total number of modes which have to be swapped, instead of
    # arbitrarily preserving the order of the modes in the first tensor. There should be many possible minor
    # optimizations, like sticking to lists instead of converting between lists and strings. 
    def conv_index_perm(input_subscript, convolution_subscript, enforce_convolution_order=False):
        
        conv_positions = occurrence_indices(input_subscript, convolution_subscript)
        
        #print("conv_positions1 = " + str(conv_positions1) + "\n")
        conv_positions_perm = []
        conv_perm = [-1]*len(input_subscript) 
        if enforce_convolution_order:
            conv_positions_perm = [-1]*len(convolution_subscript) 

            # the convolution subscript in the order it appears in the input_subscript
            input_convolution_subscript = ''.join(input_subscript[i] for i in conv_positions)

            ind = 0            
            for s in convolution_subscript:
                conv_positions_perm[ind] = input_convolution_subscript.index(s)
                ind += 1

            for i in range(0, len(conv_positions)):
                conv_perm[conv_positions[conv_positions_perm[i]]] = len(input_subscript) - len(conv_positions) + i
        else:
            for i in range(0, len(conv_positions)):
                conv_perm[conv_positions[i]] = len(input_subscript) - len(conv_positions) + i
    
        cur_non_conv_ind = 0
        for i in range(0, len(conv_perm)):
            if(conv_perm[i] == -1):
                conv_perm[i] = cur_non_conv_ind
                cur_non_conv_ind += 1
        
        return conv_perm, conv_positions

    conv_index_perm0, conv_positions0 = conv_index_perm(input_subscripts[0], convolution_subscript)   
    input_subscript0_ordered_conv_indices = ''.join([input_subscripts[0][i] for i in conv_positions0])
    conv_index_perm1, _ = conv_index_perm(input_subscripts[1], input_subscript0_ordered_conv_indices, True)
  

    
            
    input_subscript0_permuted = ''.join(permuted_collection(input_subscripts[0], conv_index_perm0))
    input_subscript1_permuted = ''.join(permuted_collection(input_subscripts[1], conv_index_perm1))
    
    operand0_permuted = operands[0].permute(conv_index_perm0)
    operand1_permuted = operands[1].permute(conv_index_perm1)

    print("operand0 = \n" + str(operands[0]))
    print("operand0_permuted = \n" + str(operand0_permuted) + "\n")

    print("operand1 = \n" + str(operands[1]))
    print("operand1_permuted = \n" + str(operand1_permuted) + "\n")
    
  
    print("convolution_subscript = " + convolution_subscript)
    print("conv_index_perm0 = " + str(conv_index_perm0))
    print("input_subscript0 = " + str(input_subscripts[0]) + " input_subscript0_permuted " + input_subscript0_permuted + "\n")
    
    print("conv_index_perm1 = " + str(conv_index_perm1))
    print("input_subscript1 = " + str(input_subscripts[1]) + " input_subscript1_permuted " + input_subscript1_permuted + "\n")

    # now we reshape the tensors so that the non-convolution indices are treated as a single index 
    last_non_conv_index0 = len(input_subscripts[0]) - len(convolution_subscript) - 1
    last_non_conv_index1 = len(input_subscripts[1]) - len(convolution_subscript) - 1
    operand0_permuted_flattened = torch.flatten(operand0_permuted, start_dim=0, end_dim=last_non_conv_index0)
    operand1_permuted_flattened = torch.flatten(operand1_permuted, start_dim=0, end_dim=last_non_conv_index1)
    
    print("operand0_permuted_flattened.size() = " + str(operand0_permuted_flattened.size()))
    print("operand1_permuted_flattened.size() = " + str(operand1_permuted_flattened.size()))




def conv_einsum_pair(*operands):
    
    # first evaluate all indices which don't appear as outputs and which appear in only one of the two tensors
    input_subscripts, output_subscript, convolution_subscript, subscripts_set, operands \
        = _parse_conv_einsum_input(operands) 
    
    input_subscript0_set = set(input_subscripts[0])
    input_subscript1_set = set(input_subscripts[1])
    output_subscript_set = set(output_subscript)

    #non_outputs_appear_once0 = list(input_subscript0_set - input_subscript1_set - output_subscript_set)
    #non_outputs_appear_once1 = list(input_subscript1_set - input_subscript0_set - output_subscript_set)

    non_outputs_appear_once0 = [e for e in input_subscripts[0] if (e not in input_subscripts[1] and e not in output_subscript)]
    non_outputs_appear_once1 = [e for e in input_subscripts[1] if (e not in input_subscripts[0] and e not in output_subscript)]
    #print(non_outputs_appear_once0)
    #print(non_outputs_appear_once1)
    

    # \todo hopefully set maintains the order of first appearance, or has some convention, but may want to enforce one if not
    appear_once0_output_subscript = ''.join([e for e in input_subscripts[0] if e not in non_outputs_appear_once0])
    appear_once1_output_subscript = ''.join([e for e in input_subscripts[1] if e not in non_outputs_appear_once1])

    appear_once0_einsum_str = '->'.join([input_subscripts[0], appear_once0_output_subscript])
    appear_once1_einsum_str = '->'.join([input_subscripts[1], appear_once1_output_subscript])

    #print(appear_once0_einsum_str)
    #print(appear_once1_einsum_str)

    #print(operands[0])
    #print(operands[1])
    summed_out_tensor0 = torch.einsum(appear_once0_einsum_str, operands[0])
    summed_out_tensor1 = torch.einsum(appear_once1_einsum_str, operands[1])

    #print(summed_out_tensor0)
    #print(summed_out_tensor1)

    # derive the conv_einsum string corresponding to the convolution part of the computation

    # \todo make sure this doesn't reorder things unnecessarily
    conv_only_output_subscript = ''.join([e for e in appear_once0_output_subscript if e in appear_once1_output_subscript])
    remaining_conv_only_subscript = ''.join([e for e in output_subscript if e not in conv_only_output_subscript])
    conv_only_output_subscript = ''.join([conv_only_output_subscript, remaining_conv_only_subscript])

    conv_only_einsum_str = ','.join([appear_once0_output_subscript, appear_once1_output_subscript])
    conv_only_einsum_str = '->'.join([conv_only_einsum_str, conv_only_output_subscript]) 
    conv_only_einsum_str = '|'.join([conv_only_einsum_str, convolution_subscript])

    print("conv_only_einsum_str: " + conv_only_einsum_str)
    
    # then compute the convolution-only part of the computation
    convolved_tensor = conv_einsum_pair_convolution_only(conv_only_einsum_str, summed_out_tensor0, summed_out_tensor1)
    

    # and derive a normal einsum string for summing out the remaining modes, which were effectively
    # elementwise multiplied in the previous calculation 
    contract_einsum_str = '->'.join([conv_only_output_subscript, output_subscript])
    print("contract_einsum_str: " + contract_einsum_str)

    #return torch.einsum(contract_einsum_str, convolved_tensor)

#torch_A = torch.randn(3, 4, 5, device='cuda:0')   


#torch_A = torch.tensor([[1,2], [3,4], [5,6]])
#torch_B = torch.tensor([4,5,6])
#conv_einsum_pair_convolution_only("ij,i->ij | i", torch_A, torch_B)



#torch_A = torch.tensor([[[1,2], [3,4]], [[5,6], [7,8]]])
#torch_B = torch.tensor([[[11,12], [13,14]], [[15,16], [17,18]]])
#conv_einsum_pair_convolution_only("jki, kij->ijk | k", torch_A, torch_B)


#torch_A = torch.randn(2,2,2,2)
#torch_B = torch.randn(2,2)
#conv_einsum_pair("ijkl,ik->il|ik", torch_A, torch_B)
#print("\n\n\n")
#conv_einsum_pair("lijk,ik->il|ik", torch_A, torch_B)

