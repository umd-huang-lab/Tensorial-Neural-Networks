
import torch

from .parse_conv_einsum import _parse_conv_einsum_input
#from parse_conv_einsum import _parse_conv_einsum_input





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
    print("kernel_size = " + str(kernel_size))
    print("input_size = " + str(input_size))
    print("kernel.size() = " + str(kernel.size()))
    print("input.size() = " + str(input_tens.size()))
   

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
#torch_B = torch.randn(6,5,2,3,20)
#eins = ijklm_niklm_to_nijlm_bar_lm(torch_A, torch_B)
#print("ijklm_niklm_to_nijlm_bar_lm size = \n" + str(eins.size()))

#eins_forloop = ijklm_niklm_to_nijlm_bar_lm_forloop(torch_A, torch_B)
#print("ijklm_niklm_to_nijlm_bar_lm_forloop = \n" + str(eins_forloop))
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

#torch_A = torch.randn(10,4,2,3,180,1)
#torch_B = torch.randn(6,10,2,30,120,1)
#
##torch_A = torch.randn(2,2,2,2,2,1)
##torch_B = torch.randn(2,2,2,2,2,1)
#
#eins = ijklmn_oiklmn_to_oijlmn_bar_lmn(torch_A, torch_B)
#eins2d = ijklm_niklm_to_nijlm_bar_lm(torch_A[:,:,:,:,:,0], torch_B[:,:,:,:,:,0])
#print((torch.squeeze(eins) - eins2d).sum())


def convolution_atomic_operation(tens1, tens2, num_convolutions):
    # This opration expects the inputs tens1 and tens2 to be shaped/permuted according to
    # the "atomic forms" given by the following functions

    if num_convolutions == 0:
        print("Error: convolution_atomic_operation expects at least one convolution index")
    elif num_convolutions == 1:
        return ijkl_mikl_to_mijl_bar_l(tens1, tens2)
    elif num_convolutions == 2:
        return ijklm_niklm_to_nijlm_bar_lm(tens1, tens2)
    elif num_convolutions == 3:
        return ijklmn_oiklmn_to_oijlmn_bar_lmn(tens1, tens2)
    else:
        print("convolution_atomic_operation num_convolutions >= 4 not implemented")


def occurrence_indices(collection, possible_occurrences):
    return [i for i in range(0, len(collection)) if collection[i] in possible_occurrences]

def elements(collection, occurrence_indices_list):
    return [collection[i] for i in occurrence_indices_list]

# this returns the index form of the permutation taking A to permutation_of_A, where A is indexed collection
def permutation_indices(A, permutation_of_A):
    return [x for x,_ in sorted(zip(range(0, len(permutation_of_A)), permutation_of_A), key=lambda perm_el: A.index(perm_el[1]))]


def permutation_inverse(perm):
    inverse = [0]*len(perm)

    for i, p in enumerate(perm):
        inverse[p] = i

    return inverse


def permuted_collection(collection, permutation):
    # returns list
    permuted = [0]*len(permutation) 
    for i in range(0, len(permutation)):
        permuted[permutation[i]] = collection[i]

    return permuted




def atomic_permutation(input_subscripts0, input_subscripts1, output_subscript, convolution_subscript, subscripts_set, input0_tensor_size, input1_tensor_size):
    # This computes the permutations necessary to permute the two input tensors and hypothetical output tensor into atomic form.
    # This also returns number of indices of each type: group, batch0, batch1, contraction, convolution
    # This also returns the convolution indices which are supposed to be summed out after the convolution is done

    # This assumes there are no nonoutput indices appearing in only one tensor.
    # This also assumes any convolution index appears in both tensors. A convolution index appearing in only one of the two tensors
    # should be turned into a contraction index.
    #
    # Due to the definitions of Conv1d, Conv2d, and Conv3d, we need to permute our tensors so that
    # the indices are laid out as follows:
    # (group indices, batch0 indices, contraction indices, convolution indices) x (batch1 indices, group indices, contraction indices, convolution indices)
    #  -> (batch1 indices, group indices, batch0 indices, convolution indices)
    #
    # The indices of each type must appear in the same order on both sides. That order is determined by the order they appear as outputs for output indices,
    # and the order they appear in the first tensor for nonoutput indices (which are contractions and convolutions without outputs)
    # Also, if a convolution index is not an output, then we must add it in as an output and sum it out at the end. It is added after the convolution indices which are outputs, in the
    # order it appears in the first tensor. 


    # this can surely be optimized, it is probably not a bottleneck though
    # \todo there may be a cleaner way to organize this code, also I tacked on input0_tensor_size and input1_tensor_size and the computation those are used for
    
    print("input_subscripts0 = " + input_subscripts0) 
    print("input_subscripts1 = " + input_subscripts1) 

    input_subscripts0_set = set(input_subscripts0)
    input_subscripts1_set = set(input_subscripts1)
    output_subscript_set = set(output_subscript)

    batch0_indices_set = set(subscripts_set) - input_subscripts1_set 
    batch1_indices_set = set(subscripts_set) - input_subscripts0_set 
    both_indices_set = input_subscripts0_set.intersection(input_subscripts1_set)
    convolution_indices_set = set(convolution_subscript)
    nonoutput_convolution_indices_set = convolution_indices_set - output_subscript_set
    group_indices_set = both_indices_set.intersection(output_subscript_set) - convolution_indices_set
    contraction_indices_set = both_indices_set - output_subscript_set - convolution_indices_set
    

    batch0_positions = occurrence_indices(input_subscripts0, batch0_indices_set)
    batch1_positions = occurrence_indices(input_subscripts1, batch1_indices_set)
    
    group0_positions = occurrence_indices(input_subscripts0, group_indices_set) # positions in input_subscripts0 which have group indices
    group1_positions = occurrence_indices(input_subscripts1, group_indices_set) # likewise for input_subscripts1
    contraction0_positions = occurrence_indices(input_subscripts0, contraction_indices_set)
    contraction1_positions = occurrence_indices(input_subscripts1, contraction_indices_set)
    convolution0_positions = occurrence_indices(input_subscripts0, convolution_indices_set)
    convolution1_positions = occurrence_indices(input_subscripts1, convolution_indices_set)

    nonoutput_convolution0_positions = occurrence_indices(input_subscripts0, nonoutput_convolution_indices_set) 
    output_subscript_conv_appended = output_subscript + ''.join(elements(input_subscripts0, nonoutput_convolution0_positions))

    
    convolution_out_positions = occurrence_indices(output_subscript_conv_appended, convolution_indices_set)
    batch0_out_positions = occurrence_indices(output_subscript, batch0_indices_set)
    batch1_out_positions = occurrence_indices(output_subscript, batch1_indices_set)
    group_out_positions = occurrence_indices(output_subscript, group_indices_set)

    
    batch0_perm = permutation_indices(elements(input_subscripts0, batch0_positions), elements(output_subscript, batch0_out_positions))
    batch1_perm = permutation_indices(elements(input_subscripts1, batch1_positions), elements(output_subscript, batch1_out_positions))
    group0_perm = permutation_indices(elements(input_subscripts0, group0_positions), elements(output_subscript, group_out_positions))
    group1_perm = permutation_indices(elements(input_subscripts1, group1_positions), elements(output_subscript, group_out_positions))
    contraction0_perm = list(range(0, len(contraction0_positions)))
    contraction1_perm = permutation_indices(elements(input_subscripts1, contraction1_positions), elements(input_subscripts0, contraction0_positions))
    convolution0_perm = permutation_indices(elements(input_subscripts0, convolution0_positions), elements(output_subscript_conv_appended, convolution_out_positions))    
    convolution1_perm = permutation_indices(elements(input_subscripts1, convolution1_positions), elements(output_subscript_conv_appended, convolution_out_positions))

    batch0_out_perm = list(range(0, len(batch0_out_positions)))
    batch1_out_perm = list(range(0, len(batch1_out_positions)))
    group_out_perm = list(range(0, len(group_out_positions)))
    convolution_out_perm = list(range(0, len(convolution_out_positions)))

    
    type_sizes = [len(group0_positions), len(batch0_positions), len(batch1_positions), len(contraction0_positions), len(convolution_subscript)]

    def calc_permutation(types, positions, permutations, subscript):
        permutation_size = 0 
        for t in range(0, len(types)):
            permutation_size += type_sizes[types[t]]
        perm_out = [0] * permutation_size

        type_offset = 0
        for t in range(0, len(types)):
            j = 0 
            for i in range(0, type_sizes[types[t]]): 
                perm_out[type_offset + permutations[t][i]] = positions[t][i] 

            type_offset += type_sizes[types[t]]
                

        return perm_out



    GROUP = 0
    BATCH0 = 1
    BATCH1 = 2
    CONTRACTION = 3
    CONVOLUTION = 4

    input0_positions = [group0_positions, batch0_positions, contraction0_positions, convolution0_positions]
    input0_permutations = [group0_perm, batch0_perm, contraction0_perm, convolution0_perm]
    input0_types = [GROUP, BATCH0, CONTRACTION, CONVOLUTION]
    input0_perm = calc_permutation(input0_types, input0_positions, input0_permutations, input_subscripts0)
     

    input1_positions = [batch1_positions, group1_positions, contraction1_positions, convolution1_positions]
    input1_permutations = [batch1_perm, group1_perm, contraction1_perm, convolution1_perm]
    input1_types = [BATCH1, GROUP, CONTRACTION, CONVOLUTION]
    input1_perm = calc_permutation(input1_types, input1_positions, input1_permutations, input_subscripts1)

 
    out_positions = [batch1_out_positions, group_out_positions, batch0_out_positions, convolution_out_positions] 
    out_permutations = [batch1_out_perm, group_out_perm, batch0_out_perm, convolution_out_perm] 
    out_types = [BATCH1, GROUP, BATCH0, CONVOLUTION]
    out_perm = calc_permutation(out_types, out_positions, out_permutations, output_subscript_conv_appended)


    
    batch0_total_dim = 1
    for pos in batch0_positions:
        batch0_total_dim *= input0_tensor_size[pos]

    batch1_total_dim = 1
    for pos in batch1_positions:
        batch1_total_dim *= input1_tensor_size[pos]

    group_total_dim = 1
    for pos in group0_positions:
        group_total_dim *= input0_tensor_size[pos] 

    contraction_total_dim = 1
    for pos in contraction0_positions:
        contraction_total_dim *= input0_tensor_size[pos]


    
    # \todo convolution_out_total_dim is computed incorrectly
    print("convolution0_perm = " + str(convolution0_perm))
    print("convolution1_perm = " + str(convolution1_perm))

    convolution0_perm_inverse = permutation_inverse(convolution0_perm) 
    convolution1to0_perm = [convolution0_perm_inverse[convolution1_perm[i]] for i in range(0, len(convolution1_perm))]
    print("convolution1to0_perm = " + str(convolution1to0_perm))


    convolution_out_total_dim = 1     
    convolution_out_size = []
    for i in range(0, len(convolution0_positions)):
        conv0_sz_i = input0_tensor_size[convolution0_positions[convolution0_perm[i]]]
        conv1_sz_i = input1_tensor_size[convolution1_positions[convolution1_perm[i]]] 
        convolution_out_size.append(max(conv0_sz_i, conv1_sz_i))
    

    print("\n")
    print("convolution0_positions: " + str(convolution0_positions))
    print("convolution1_positions: " + str(convolution1_positions))
    print("convolution0_perm: " + str(convolution0_perm))
    print("convolution1_perm: " + str(convolution1_perm))
    print("\n")

    
    convolution0_size =[]
    for pos in convolution0_positions:     
        convolution0_size.append(input0_tensor_size[pos])
   
    convolution1_size =[]
    for pos in convolution1_positions:     
        convolution1_size.append(input1_tensor_size[pos])
 

    reshaped0_tensor_size = [group_total_dim, batch0_total_dim, contraction_total_dim] + convolution0_size
    reshaped1_tensor_size = [batch1_total_dim, group_total_dim, contraction_total_dim] + convolution1_size
    unreshaped_out_tensor_size = [batch1_total_dim, group_total_dim, batch0_total_dim] + convolution_out_size  


    
    return input0_perm, input1_perm, out_perm, output_subscript_conv_appended, reshaped0_tensor_size, reshaped1_tensor_size, unreshaped_out_tensor_size


def conv_einsum_pair(*operands):

    input_subscripts, output_subscript, convolution_subscript, subscripts_set, operands \
        = _parse_conv_einsum_input(operands) 
     
    input_subscripts0 = input_subscripts[0]
    input_subscripts1 = input_subscripts[1]
    

    # we remove from the convolution_subscript the convolution indices appearing
    # in only one of the two tensors, because this is handled as a contraction, and
	# additionally may be summed out if it is not an output 
    
    nontrivial_convolution_subscript_list = []
    for c in convolution_subscript:  
        if (c in input_subscripts0) and (c in input_subscripts1):
            nontrivial_convolution_subscript_list += c
    convolution_subscript = ''.join(nontrivial_convolution_subscript_list)


    # If after removing nontrivial convolution indices there are no convolution indices, do the computation
    # using Pytorch's contraction-only einsum
    if len(convolution_subscript) == 0:
        standard_einsum_str = input_subscripts0 + "," + input_subscripts1 + "->" + output_subscript
        return torch.einsum(standard_einsum_str, operands[0], operands[1]) 
 

    # We evaluate all indices which don't appear as outputs and appear in only one of the two tensors
    input_subscript0_set = set(input_subscripts0)
    input_subscript1_set = set(input_subscripts1)
    output_subscript_set = set(output_subscript)

    non_outputs_appear_once0 = [e for e in input_subscripts0 if (e not in input_subscripts1 and e not in output_subscript)]
    non_outputs_appear_once1 = [e for e in input_subscripts1 if (e not in input_subscripts0 and e not in output_subscript)]
    
    appear_once0_output_subscript = ''.join([e for e in input_subscripts0 if e not in non_outputs_appear_once0])
    appear_once1_output_subscript = ''.join([e for e in input_subscripts1 if e not in non_outputs_appear_once1])

    appear_once0_einsum_str = '->'.join([input_subscripts0, appear_once0_output_subscript])
    appear_once1_einsum_str = '->'.join([input_subscripts1, appear_once1_output_subscript])

  
    if(len(non_outputs_appear_once0) > 0):
        summed0_tensor = torch.einsum(appear_once0_einsum_str, operands[0])
    else:
        summed0_tensor = operands[0]
    if(len(non_outputs_appear_once1) > 0):
        summed1_tensor = torch.einsum(appear_once1_einsum_str, operands[1])
    else:
        summed1_tensor = operands[1] # \todo is this if else block actually saving any computation?

    input_subscripts0 = appear_once0_output_subscript
    input_subscripts1 = appear_once1_output_subscript

    #print("summed_out inputs: " + str(input_subscripts0) + ", " + str(input_subscripts1))
    
    
    # we compute the data necessary for permuting/reshaping the tensors into, and from for the output, atomic form 
    input0_perm, input1_perm, out_perm, output_subscript_conv_appended, \
    reshaped0_tensor_size, reshaped1_tensor_size, unreshaped_out_tensor_size \
        = atomic_permutation(input_subscripts0, input_subscripts1, output_subscript, convolution_subscript, subscripts_set, summed0_tensor.size(), summed1_tensor.size()) 

    #print("permuted input_subscripts0 = " + ''.join(elements(input_subscripts0, input0_perm)))
    #print("permuted input_subscripts1 = " + ''.join(elements(input_subscripts1, input1_perm)))
    #print("permuted output_subscript_conv_appended = " + ''.join(elements(output_subscript_conv_appended, out_perm)))
    

    # we do the permuting and reshaping, call our atomic operation, then reshape and permute the output
   
    #print("input0_perm = " + str(input0_perm))
     
    reshaped0_tensor = summed0_tensor.permute(input0_perm).reshape(reshaped0_tensor_size)
    reshaped1_tensor = summed1_tensor.permute(input1_perm).reshape(reshaped1_tensor_size)

    print("reshaped0_tensor_size = " + str(reshaped0_tensor_size))
    print("reshaped1_tensor_size = " + str(reshaped1_tensor_size))
    print("reshaped0_tensor.size() = " + str(reshaped0_tensor.size()))
    print("reshaped1_tensor.size() = " + str(reshaped1_tensor.size()))

    num_conv = len(convolution_subscript)
    
    #print("reshaped0_tensor.size() = " + str(reshaped0_tensor.size()))
    unreshaped_out = convolution_atomic_operation(reshaped0_tensor, reshaped1_tensor, num_conv)

    #print("unreshaped_out_tensor_size = " + str(unreshaped_out_tensor_size)) 
    #print("unreshaped_out.size() = " + str(unreshaped_out.size()))
    print("out_perm = " + str(out_perm))
    # \todo I don't think I can simply call torch.squeeze because the 1 may actually be intended, 
    #       This means unreshaped_out_tensor_size is actually computed incorrectly
    return torch.squeeze(unreshaped_out.reshape(unreshaped_out_tensor_size)).permute(permutation_inverse(out_perm))
    #return unreshaped_out.reshape(unreshaped_out_tensor_size).permute(permutation_inverse(out_perm))

    

#torch_A = torch.randn(3, 4, 5, device='cuda:0')   


#torch_A = torch.tensor([[1,2], [3,4], [5,6]])
#torch_B = torch.tensor([4,5,6])
#conv_einsum_pair_convolution_only("ij,i->ij | i", torch_A, torch_B)



#torch_A = torch.tensor([[[1,2], [3,4]], [[5,6], [7,8]]])
#torch_B = torch.tensor([[[11,12], [13,14]], [[15,16], [17,18]]])
#conv_einsum_pair_convolution_only("jki, kij->ijk | k", torch_A, torch_B)



#torch_A = torch.ones(2,2,2,2)
#torch_B = torch.ones(2,2)
#conv_einsum_pair("ijkl,ik->il|ik", torch_A, torch_B)
#print("\n\n\n")
#conv_einsum_pair("lijk,ik->il|ik", torch_A, torch_B)

#torch_A = torch.ones(2,2,2,2,2,2,2)
#torch_B = torch.ones(2,2,2,2,2,2)
#einsum_str = "abcdefh, dafgbh -> gfdaec | fdh"
#print("einsum_str = " + einsum_str)
#conv_einsum_pair(einsum_str, torch_A, torch_B)


#torch_A = torch.ones(2,3,2)
#torch_B = torch.ones(2,2,3,4)
#einsum_str = "ijk, ijmh -> ihj | j"
#print(einsum_str + " = \n" + str(conv_einsum_pair(einsum_str, torch_A, torch_B)))


#torch_A = torch.ones(2,3)
#torch_B = torch.ones(2,2,4)
#einsum_str = "ij, ijh -> ihj | j"
#print(einsum_str + " = \n" + str(conv_einsum_pair(einsum_str, torch_A, torch_B)))


#torch_A = torch.randn(4,2)
#torch_B = torch.randn(4,20)
#einsum_str = "ij, ij -> ij | j"
#print(einsum_str + " = \n" + str(conv_einsum_pair(einsum_str, torch_A, torch_B)))
#print("ij_ij_to_ij_bar_j = \n" + str(ij_ij_to_ij_bar_j(torch_A, torch_B)))


def conv_einsum(*operands):
    input_subscripts, output_subscript, convolution_subscript, subscripts_set, operands \
        = _parse_conv_einsum_input(operands)

    # this simple pairwise reduction evaluates from left to right, and at each pair
    # it sets the output to be those indices appearing in the final output, and it sets
    # the convolution subscript to be those indices appearing in the total convolution subscript

    out = operands[0] # I'm pretty sure out is a reference, and this = doesn't do a copy
    left_input_subscript = input_subscripts[0]

    for i in range(1, len(operands)):
        right_input_subscript = input_subscripts[i]
        pair_output_subscript = []
        for s in output_subscript:
            if s in left_input_subscript or s in right_input_subscript:
                pair_output_subscript.append(s)
        pair_output_subscript = ''.join(pair_output_subscript)

        pair_str = left_input_subscript + ", " + right_input_subscript + " -> " \
                                        + pair_output_subscript + " | " + convolution_subscript
        # I think it might be better to parse the pair convolution_subscript, and not
        # pass the total convolution subscript, but this is convenient for now

        left_input_subscript = pair_output_subscript

        out = conv_einsum_pair(pair_str, out, operands[i])

    return out

#torch_A = torch.ones(4)
#torch_B = torch_A
#torch_C = torch_A
#einsum_str = "i,i,i -> i | i"
#print(einsum_str + " = \n" + str(conv_einsum(einsum_str, torch_A, torch_B, torch_C)))


#torch_A = torch.ones(4)
#torch_B = torch.ones(4,5)
#torch_C = torch.ones(4,5,6)
#einsum_str = "i,ij,ijk -> ij | ij"
#print(einsum_str + " = \n" + str(conv_einsum(einsum_str, torch_A, torch_B, torch_C)))

