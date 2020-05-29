
import torch

from parse_conv_einsum import _parse_conv_einsum_input
#from parse_conv_einsum import *




#torch_A = torch.tensor([[1,2], [3,4], [5,6]])
#torch_B = torch.tensor([4,5,6])
#conv_einsum_pair_convolution_only("ij,i->ij | i", torch_A, torch_B)
#conv_einsum_pair_convolution_only("ij,ij,i,k,k->ij | ijk", torch_A, torch_A, torch_B, torch_B, torch_B)


#torch_A = torch.randn(3, 4, 5, device='cuda:0')


#torch_A = torch.randn(2,2,2,2)
#torch_B = torch.randn(2,2)
#conv_einsum_pair("ijkl,ik->il|ik", torch_A, torch_B)



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
    # l1 denotes the shorter dim, l2 denotes the longer dim \todo
    # this padding is chosen so that l2 + 2*padding - l1 + 1 >= l2, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    # for some reason the answer is too short in the case l1 == 2, so I manually set the padding to 1... sometimes however, like when
    # l1 = l2 = 2 this causes the output vector to be too long, so I truncate the output vector
    #if(dim == 2):
    #        return 1
    #return (dim-1)//2
    #return min_size//2
    return (max(ker_size, input_size) - input_size + ker_size)//2

def padding_2d(ker_size, input_size):
    #h = (max(ker_size[0], input_size[0]) - input_size[0] + ker_size[0])//2
    #w = (max(ker_size[1], input_size[1]) - input_size[1] + ker_size[1])//2
    #return (h,w) 
    return (padding_1d(ker_size[0], input_size[0]), padding_1d(ker_size[1], input_size[1]))



def convolve_same(vec1, vec2):
    # this function should agree with the convolve functions offered by numpy and scipy, with mode='same' 
    kernel_size = len(vec1)
    input_size = len(vec2)
 
    if(kernel_size > input_size): # Conv1d requires the kernel to be shorter than the input
        return convolve_same(vec2, vec1)
  
    in_channels = 1
    out_channels = 1
    groups = 1
    
    padding = padding_1d(kernel_size, input_size)   
    
    # \todo I'm not sure on which inputs the output vector will have the right length so I'm just defensively truncating it
    
    
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    m.weight.data[0, 0, :] = torch.flip(vec1, [0])

    output = m(vec2.view(1, 1, input_size))
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

    num_convolutions = mat1.size(0) 
    if(num_convolutions != mat2.size(0)):
        print("Error: The inputs must have the same number of rows")

    kernel_size = mat1.size(1)
    input_size = mat2.size(1)
 
    if(kernel_size > input_size): # Conv1d requires the kernel to be shorter than the input
        return convolve_same(mat2, mat1)
 
    in_channels = num_convolutions
    out_channels = num_convolutions
    groups = num_convolutions

    # this padding is chosen so that input_size + 2*padding - kernel_size + 1 >= input_size, the left expression in this equation is the length of the
    # output and and is obtained by examining the general formula given for the output length from the Conv1d documentation
    padding = padding_1d(kernel_size, input_size) 
  
    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    mat1_flipped = torch.flip(mat1, [1])
    m.weight.data = mat1_flipped.view(out_channels, in_channels//groups, kernel_size)

    output = m(mat2.view(1, in_channels, input_size))

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








def independent_convolve2d_same(in1, in2):
    
    num_convolutions = in1.size(0)
    if(num_convolutions != in2.size(0)):        
        print("Error: the inputs must have the same length in the 0th mode")

    kernel_size = in1.size()[1:3]
    input_size = in2.size()[1:3]
 
    max_h = max(kernel_size[0], input_size[0])
    max_w = max(kernel_size[1], input_size[1])
    
 
    in_channels = num_convolutions
    out_channels = num_convolutions
    groups = num_convolutions

   
    padding = padding_2d(kernel_size, input_size)
    
    m = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
    in1_flipped = torch.flip(in1, [1,2])

    m.weight.data = in1_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])

    output = m(in2.view(1, in_channels, input_size[0], input_size[1]))

    # output.data is a 1 x 1 x width x height tensor
    # and we slice it to the proper dimensions, because Conv2d returns too many
    # rows and columns with the best possible padding
    return torch.squeeze(output.data[:,:,:max_h,:max_w], 0) 




B = [[[1., 2., 3.]], [[1.,1.,1.]]]
A = [[[2., -3.], [14., 24.]], [[1.,1.], [1., 1.]]]

torch_A = torch.tensor(A)
torch_B = torch.tensor(B)

print("torch_A = \n" + str(torch_A))
print("torch_B = \n" + str(torch_B))
print("independent_convolve2d_same = \n" + str(independent_convolve2d_same(torch_A, torch_B)) + "\n")

print("single1: " + str(torch_A[0,:,:]))
print("single2: " + str(torch_B[0,:,:]))
print("single = \n" + str(convolve2d_same(torch_A[0,:,:], torch_B[0,:,:])) + "\n")





# this function computes the special case of conv_einsums where the only operation is N-dimensional convolution
def conv_einsum_pair_convolution_only(*operands):
    # \type This function assumes the einsum string is of a particular type
    input_subscripts, output_subscript, convolution_subscript, subscripts_set, operands \
        = _parse_conv_einsum_input(operands) 

    # \todo just assuming there's only a single convolution for now, will add support for N dimensional convolutions
    # actually, I'll try to implement 2 and 3 way convolutions first, because that will give me a better idea for what
    # to expect when implementing this method

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
    ##convolved_tensor = conv_einsum_pair_convolution_only(conv_only_einsum_str, summed_out_tensor0, summed_out_tensor1)
    convolved_tensor = []

    # and derive a normal einsum string for summing out the remaining modes, which were effectively
    # elementwise multiplied in the previous calculation 
    contract_einsum_str = '->'.join([conv_only_output_subscript, output_subscript])
    print(contract_einsum_str)

    return torch.einsum(contract_einsum_str, convolved_tensor)

    

