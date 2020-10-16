
import torch

#from .parse_conv_einsum import _parse_conv_einsum_input
from parse_conv_einsum import _parse_conv_einsum_input

from collections import OrderedDict
from collections.abc import Mapping

import math



def padding_mode_zeros_L_out(kernel_size, input_size, padding, stride, dilation):
    # This is from the documentation for Conv1d
    # It returns the expected output length for a given convolution with 
    # padding_mode='zeros'
    return math.floor(input_size + (2*padding - dilation*(kernel_size-1) - 1)/stride + 1)


###
# This computes the zero padding to append to the input image vector so that
# the resulting convolution has the same length as the maximum length of the kernel
# and the input tensor. Note the kernel can be longer than the input tensor. Additionally,
# conv_einsum("i, i -> i | i", A, B, padding=) = conv_einsum("i, i -> i | i", B, A, padding=)
# when this padding is selecting. These two properties may not hold when dilation > 1.
# \todo
#
def max_zeros_padding_1d(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):     
    # see https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html for the definition
    # under the Shape section for the definition of the padding. We add 1 so the integer division
    # by 2 errs large instead of small
    # for 1d convolutions ker_mode_size < input_mode_size but this is not true for higher 
    # dimensions, and I use this function to compute those paddings 

    # "input" tensor is synonymous with "image" tensor in much of this
 
    max_ker_input = max(ker_mode_size, input_mode_size, max_mode_size)
    
    # \todo it's not clear if this works if stride doesn't divide evenly into
    #           input_mode_size + 2*padding - dilation * (kernel_size - 1) - 1
    #       (see Conv1d documentation)
    #       It does however appear to work whenever dilation = 1, or when
    #       dilation*kernel_size < input_size + some factor involving stride
    # padding can only be negative if kernel_mode_size == 0 
    twice_padding = (max_ker_input-1)*stride + 1 + dilation*(ker_mode_size-1) - input_mode_size 

    return (twice_padding+1)//2 # add 1 so its never less than twice_padding/2
    

def max_zeros_padding_2d(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):     
    return (max_zeros_padding_1d(ker_mode_size[0], input_mode_size[0], \
                                 max_mode_size[0], stride[0], dilation[0]), \
            max_zeros_padding_1d(ker_mode_size[1], input_mode_size[1], \
                                 max_mode_size[1], stride[1], dilation[1])) 

def max_zeros_padding_3d(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):     

    return (max_zeros_padding_1d(ker_mode_size[0], input_mode_size[0], \
                                 max_mode_size[0], stride, dilation), \
            max_zeros_padding_1d(ker_mode_size[1], input_mode_size[1], \
                                 max_mode_size[1], stride[1], dilation[1]), 
            max_zeros_padding_1d(ker_mode_size[2], input_mode_size[2], \
                                 max_mode_size[2], stride[2], dilation[2])) 


def max_zeros_padding_nd(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1): 
    return tuple(max_zeros_padding_1d(ker_mode_size[i], input_mode_size[i], \
                                   max_mode_size[i], stride[i], dilation[i]) \
                 for i in range(0,len(ker_mode_size)))


# \todo this function is not fully implemented
#       I'm not sure if we really need to support nontrivial hyperparameter cases though
# \todo How to tell the user to specify the maximum mode size of an N way convolution
#        to input_mode_size
def max_circular_padding_1d(ker_mode_size, input_mode_size, \
                            stride=1, dilation=1): 

  
    # the formulas in this function were determined by experimentally computing the first pad
    # which did not cause a run time error
    # \todo These formulas should be justified rigorously 
    if stride != 1:
        # the padding seems to be more complicated in this case
        print("Error: stride != 1, case not implemented")

    if dilation == 1:
        ker_img_ratio = (ker_mode_size + 1) / (input_mode_size)

        if ker_img_ratio >= 2:
            print("Error: stride == 1, dilation == 1, ker+1 >= 2*img \
                   implies no pad is possible, and \
                   proper handling of this case is not yet implemented")
        elif 1 < ker_img_ratio and ker_img_ratio < 2:
            return 2*ker_mode_size - input_mode_size - 1
        else:
            return ker_mode_size
    else:
        # \todo This formula is not actually correct (at least for dilation == 6)...
        #       I'm not sure if this is a bug with pytorch (meaning no possible formula
        #       will work), or if there's another factor I'm missing
         if ker_mode_size > input_mode_size:
             print("Error: dilation > 2 and ker_mode_size > input_mode_size \
                    implies no pad is possible, and \
                    proper handling of this case is not yet implemented")
         else:
             return dilation*(ker_mode_size-1)





def atomic_pad(tensor, padding):

    if len(tensor.size()) == 4:
        # ijkl_mikl_to_mijl_bar_l   ("ijkl, mikl -> mijl | l")
        # in this case we have to append 0s to the last mode
        #tensor = torch.cat(torch.cat(tensor
        print("temp")
    elif len(tensor.size()) == 5:
        # ijklm_niklm_to_nijlm_bar_lm    ("ijklm, niklm -> nijlm | lm")    
        print("temp")
    elif len(tensor.size()) == 6:
        # ijklmn_oiklmn_to_oijlmn_bar_lmn   ("ijklmn, oiklmn -> oijlmn | lmn")
        print("temp")
    else:
        print("Error: atomic_circular_pad tensor order not implemented")



# \todo it's actually easier to read if I change this to
#       mikl_ijkl_to_mijl_bar_l
#       which is the same as
#       ijkl_jmkl_to_ijml_bar_l # is this right?
#       because then the batch indices appear in the right order
# \todo I should probably switch the order of the inputs to stick to convention
# Supports padding_mode - 'max_zeros', 'max_circular', 'zeros', 'reflect', 'replicate' or 'circular'
# If 'max_zeros' or 'max_circular' is chosen, then whatever is passed to padding will be ignored
def ijkl_mikl_to_mijl_bar_l(kernel, input_tens, max_mode_size=None, \
                            padding_mode='zeros', padding=0, stride=1, dilation=1, bias=False):
    # this is supposed to compute the most general, up to reshaping / permuting, 
    # convolutional einsum with 1 convolution index which is computable by Conv1d alone.
    # This is the order the indices must appear in so that the operation can be done 
    # without any calls to permute.

    # By the Conv1d documentation https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
    # input = (batch_size = M, input_channel = groups*K = I*K, input_size = input_tens.size(3))
    # weight = (out_channels = groups * J = I * J, in_channels/groups = K, kernel_size = kernel.size(3))
    # output = (batch_size = M, out_channels = groups * J = I * J, conv_len = max(input_size, kernel_size))
    # and it appears the indices are laid out as (so, after reshaping)
    # input: M, I, K, L
    # weight: I, J, K, L
    # output: M, I, J, L

    batch_size = input_tens.size(0)
    kernel_size = kernel.size(3)

    input_size = input_tens.size(3) # image_size is perhaps a better name
    conv_len = max(kernel_size, input_size) 
    if max_mode_size != None:
        conv_len = max(conv_len, max_mode_size)# \todo
   
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)
  

    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, \
                        kernel_size=kernel_size, \
                        stride=stride, padding=padding, padding_mode=padding_mode, \
                        dilation=dilation, \
                        groups=groups, bias=bias)
    kernel_flipped = torch.flip(kernel, [3])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size)
    
    output = m(input_tens.reshape(batch_size, in_channels, input_size))  
    out_shape = (batch_size, groups, out_channels//groups, conv_len//stride) 
   
    # \todo Cutting the mode down only makes sense for a max padding 
    #       or if they're passing the maximum expected size
    return torch.reshape(output.data[:,:,:(conv_len//stride)], out_shape)
    



#I = 1
#J = 1
#K = 1
#L_kernel = 7
#L_input = 13
#M = 1
#
#A = torch.ones(I, J, K, L_kernel)
#B = torch.ones(M, I, K, L_input)
#
#
#max_mode_size = None
#stride = 1
#dilation = 1
##padding = max_zeros_padding_1d(L_kernel, L_input, \
##                               max_mode_size=max_mode_size, \
##                               stride=stride, dilation=dilation) 
#padding = max_circular_padding_1d(L_kernel, L_input)
#
#T = ijkl_mikl_to_mijl_bar_l(A, B, padding_mode='circular', padding=padding)
#
#print(str(T.size()))


def ijklm_niklm_to_nijlm_bar_lm(kernel, input_tens, max_mode_sizes=None, \
                               padding_mode='zeros', \
                               padding=0, stride=1, dilation=1, bias=False):

    # This is supposed to compute the most general, up to reshaping / permuting, 
    # convolutional einsum with 2 convolution indices which is computable by Conv2d alone.
    
    batch_size = input_tens.size(0)
    kernel_size = kernel.size()[3:5]
    input_size = input_tens.size()[3:5]
    
    max_h = max(kernel_size[0], input_size[0])
    max_w = max(kernel_size[1], input_size[1]) 
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)

 
    m = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                        kernel_size=kernel_size, stride=stride, \
                        padding=padding, padding_mode=padding_mode, \
                        dilation=dilation, groups=groups, bias=bias)

    kernel_flipped = torch.flip(kernel, [3,4])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])
    
    output = m(input_tens.reshape(batch_size, in_channels, input_size[0], input_size[1]))
    
    try:
        stride[0]
    except TypeError:
        stride = [stride]*2

    out_shape = (batch_size, groups, out_channels//groups, max_h//stride[0], max_w//stride[1]) 
    return torch.reshape(output.data[:,:,:(max_h//stride[0]),:(max_w//stride[1])], out_shape)



def ijklmn_oiklmn_to_oijlmn_bar_lmn(kernel, input_tens, max_mode_sizes=0, \
                                    padding_mode='zeros', \
                                    padding=0, stride=1, dilation=1, bias=False):
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
 

    m = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
                        stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation, groups=groups, bias=bias)

    kernel_flipped = torch.flip(kernel, [3,4,5])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2])
    
    output = m(input_tens.reshape(batch_size, in_channels, input_size[0], input_size[1], input_size[2]))
    
    try:
        stride[0]
    except TypeError:
        stride = [stride]*3

    out_shape = (batch_size, groups, out_channels//groups, max_d//stride[0], max_h//stride[1], max_w//stride[2]) 
    return torch.reshape(output.data[:,:,:(max_d//stride[0]),:(max_h//stride[1]),:(max_w//stride[2])], out_shape)



# \todo Should swap the order of kernel / input_tens to adhere to convention
# \todo I want these functions to not do any special padding. That should be done in
#       a higher level function, so 
#
def convolution_atomic_operation(kernel, input_tens, num_convolutions, max_mode_sizes, \
                                 padding_mode='max_zeros', padding=0, stride=1, dilation=1, \
                                 bias=False):
    # This operation expects the inputs input_tens and kernel to be shaped/permuted according to
    # the "atomic forms" given by the following functions
    # note the convolution indices always appear at the end

    
    if padding_mode == 'max_zeros':
        kernel_size = kernel.size()[3:len(kernel.size())]
        input_size = input_tens.size()[3:len(input_tens.size())] 
        padding = max_zeros_padding_nd(kernel_size, input_size, \
                                       max_mode_size=max_mode_sizes, \
                                       stride=stride, dilation=dilation)
        
        padding_mode = 'zeros'
    elif padding_mode == 'max_circular':
        #padding = ??
        print("padding_mode == max_circular not implemented")
        padding_mode = 'circular'   

    if num_convolutions == 0:
        print("Error: convolution_atomic_operation expects at least one convolution index")
    elif num_convolutions == 1: 
        stride = stride[0]
        dilation = dilation[0]
        padding = padding[0]
        max_mode_size = max_mode_sizes[0] 
 
        return ijkl_mikl_to_mijl_bar_l(kernel, input_tens, max_mode_size=max_mode_size, \
                                       padding_mode=padding_mode, padding=padding, \
                                       stride=stride, dilation=dilation, bias=bias)
    elif num_convolutions == 2:
        return ijklm_niklm_to_nijlm_bar_lm(kernel, input_tens, max_mode_sizes=max_mode_sizes, \
                                           padding_mode=padding_mode, padding=padding, \
                                           stride=stride, dilation=dilation, bias=bias)
    elif num_convolutions == 3:
        return ijklmn_oiklmn_to_oijlmn_bar_lmn(kernel, input_tens, \
                                               max_mode_sizes=max_mode_sizes, \
                                               padding_mode=padding_mode, padding=padding, \
                                               stride=stride, dilation=dilation, bias=bias)
    else:
        print("convolution_atomic_operation num_convolutions >= 4 not implemented")



def occurrence_indices(collection, possible_occurrences):
    return [i for i in range(0, len(collection)) if collection[i] in possible_occurrences]

def elements(collection, occurrence_indices_list):
    return [collection[i] for i in occurrence_indices_list]

# this returns the index form of the permutation taking the corresponding subcollection of A to 
# permutation_of_A_subcollection, where A is indexed collection and permutation_of_A_subcollection is some ordered subcollection of A
# this function can also be used like an ordered_occurrence_indices, except the result is always a permutatation (so it gives the
# relative ordered of the subcollection, not the order of the subcollection in the bigger collection)
def permutation_indices(A, permutation_of_A_subcollection):
    return [x for x,_ in sorted(zip(range(0, len(permutation_of_A_subcollection)), permutation_of_A_subcollection), key=lambda perm_el: A.index(perm_el[1]))]


def permuted_collection(collection, permutation):
    # returns list
    permuted = [0]*len(permutation) 
    for i in range(0, len(permutation)):
        permuted[permutation[i]] = collection[i]

    return permuted

def ordered_occurrence_indices(collection, possible_occurrences):
    occ = occurrence_indices(collection, possible_occurrences)
    perm = permutation_indices(collection, possible_occurrences)

    return permuted_collection(occ, perm)


def permutation_inverse(perm):
    inverse = [0]*len(perm)

    for i, p in enumerate(perm):
        inverse[p] = i

    return inverse






def atomic_permutation(input_subscripts0, input_subscripts1, output_subscript_conv_appended, \
                       convolution_subscript, indices_set, \
                       input0_tensor_size, input1_tensor_size, \
                       stride_tuple, max_mode_sizes_tuple):
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
    

    input_subscripts0_set = set(input_subscripts0)
    input_subscripts1_set = set(input_subscripts1)
    output_subscript_set = set(output_subscript_conv_appended)

    batch0_indices_set = set(indices_set) - input_subscripts1_set 
    batch1_indices_set = set(indices_set) - input_subscripts0_set 
    both_indices_set = input_subscripts0_set.intersection(input_subscripts1_set)
    convolution_indices_set = set(convolution_subscript) 
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

    
    convolution_out_positions = occurrence_indices(output_subscript_conv_appended, convolution_indices_set)
    batch0_out_positions = occurrence_indices(output_subscript_conv_appended, batch0_indices_set)
    batch1_out_positions = occurrence_indices(output_subscript_conv_appended, batch1_indices_set)
    group_out_positions = occurrence_indices(output_subscript_conv_appended, group_indices_set)
   
      
    # \todo why is the permutation for the inputs using the output positions?
    # surely these are computed incorrectly...
    # If we permute the batch inputs so that they appear in the same order as they do in the output, then the atomic operation computes
    # them in the order that they will appear in the final output, so no additional permuting is required
    batch0_perm = permutation_indices(elements(input_subscripts0, batch0_positions), elements(output_subscript_conv_appended, batch0_out_positions))
    batch1_perm = permutation_indices(elements(input_subscripts1, batch1_positions), elements(output_subscript_conv_appended, batch1_out_positions))
    # we permute both group indices for both inputs so they appear in the same order as in the output. This makes them consistently aligned
    # so then after reshaping a single group operation does the correct independent group operations. Aligning them to the output has the
    # benefit that no further permuting is necessary.
    group0_perm = permutation_indices(elements(input_subscripts0, group0_positions), elements(output_subscript_conv_appended, group_out_positions))
    group1_perm = permutation_indices(elements(input_subscripts1, group1_positions), elements(output_subscript_conv_appended, group_out_positions))
    # contraction has no output index, so in order to consistently align the contraction indices we permute the second input's contactions so
    # that they appear in the same order as the first input's
    contraction0_perm = list(range(0, len(contraction0_positions)))
    contraction1_perm = permutation_indices(elements(input_subscripts1, contraction1_positions), elements(input_subscripts0, contraction0_positions))
    # as with the group indices, we consistently align the convolution to the output 
    convolution0_perm = permutation_indices(elements(input_subscripts0, convolution0_positions), elements(output_subscript_conv_appended, convolution_out_positions))    
    convolution1_perm = permutation_indices(elements(input_subscripts1, convolution1_positions), elements(output_subscript_conv_appended, convolution_out_positions))
    
    # calc_permutation combines the permutations for each type into one permutation
    type_sizes = [len(group0_positions), len(batch0_positions), len(batch1_positions), len(contraction0_positions), len(convolution_subscript)] 
    def calc_permutation(types, positions, permutations):
        # this is confusing because types and type_sizes are indexed differently... type_sizes is index by the elements of types
        
        permutation_size = 0 
        for t in range(0, len(types)):
            permutation_size += type_sizes[types[t]]
        perm_out_inverse = [0] * permutation_size

        type_offset = 0
        for t_index in range(0, len(types)): 
            typ = types[t_index]
           
            for i in range(0, type_sizes[typ]): 
                perm_out_inverse[type_offset + permutations[t_index][i]] = positions[t_index][i] 

            type_offset += type_sizes[typ] 

        return permutation_inverse(perm_out_inverse)



    GROUP = 0
    BATCH0 = 1
    BATCH1 = 2
    CONTRACTION = 3
    CONVOLUTION = 4

    input0_positions = [group0_positions, batch0_positions, contraction0_positions, convolution0_positions]
    input0_permutations = [group0_perm, batch0_perm, contraction0_perm, convolution0_perm]
    input0_types = [GROUP, BATCH0, CONTRACTION, CONVOLUTION]
    input0_perm = calc_permutation(input0_types, input0_positions, input0_permutations)   

    input1_positions = [batch1_positions, group1_positions, contraction1_positions, convolution1_positions]
    input1_permutations = [batch1_perm, group1_perm, contraction1_perm, convolution1_perm]
    input1_types = [BATCH1, GROUP, CONTRACTION, CONVOLUTION]
    input1_perm = calc_permutation(input1_types, input1_positions, input1_permutations)

   
    
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
    

    batch0_out_size = []
    batch0_perm_inverse = permutation_inverse(batch0_perm)   
    for i in range(0, len(batch0_out_positions)):
        batch0_out_size += [input0_tensor_size[batch0_positions[batch0_perm_inverse[i]]]]  

    batch1_out_size = []
    batch1_perm_inverse = permutation_inverse(batch1_perm)
    for i in range(0, len(batch1_out_positions)):
        batch1_out_size += [input1_tensor_size[batch1_positions[batch1_perm_inverse[i]]]] 

    group_out_size = []
    group0_perm_inverse = permutation_inverse(group0_perm)
    for i in range(0, len(group_out_positions)):
        group_out_size += [input0_tensor_size[group0_positions[group0_perm_inverse[i]]]] 

    
    convolution0_perm_inverse = permutation_inverse(convolution0_perm) 
    convolution1_perm_inverse = permutation_inverse(convolution1_perm)
    convolution1to0_perm = [convolution0_perm_inverse[convolution1_perm[i]] for i in range(0, len(convolution1_perm))]
    convolution_out_size = []
    for i in range(0, len(convolution0_positions)):
        conv0_sz_i = input0_tensor_size[convolution0_positions[convolution0_perm_inverse[i]]]
        conv1_sz_i = input1_tensor_size[convolution1_positions[convolution1_perm_inverse[i]]] 
        if(isinstance(max_mode_sizes_tuple, tuple)):
            out_sz =  max(conv0_sz_i, conv1_sz_i, max_mode_sizes_tuple[i])//stride_tuple[i]
        else:
            out_sz =  max(conv0_sz_i, conv1_sz_i)//stride_tuple[i]
        convolution_out_size.append(out_sz)

    
    convolution0_size = []
    for i in range(0, len(convolution0_positions)):     
        pos = convolution0_positions[convolution0_perm_inverse[i]]
        convolution0_size.append(input0_tensor_size[pos])
  
    convolution1_size = []
    for i in range(0, len(convolution1_positions)):     
        pos = convolution1_positions[convolution1_perm_inverse[i]]
        convolution1_size.append(input1_tensor_size[pos])

    preatomic0_tensor_size = [group_total_dim, batch0_total_dim, contraction_total_dim] + convolution0_size
    preatomic1_tensor_size = [batch1_total_dim, group_total_dim, contraction_total_dim] + convolution1_size 
    
    out_perm = [] 
    out_types = [] 
    reshaped_out_tensor_size = []

    if(len(batch1_out_positions) > 0): # batch1 
        reshaped_out_tensor_size += batch1_out_size
        out_types += [BATCH1]
        out_perm += batch1_out_positions

    if(len(group_out_positions) > 0): # group
        reshaped_out_tensor_size += group_out_size
        out_types += [GROUP]
        out_perm += group_out_positions

    if(len(batch0_out_positions) > 0): # batch0
        reshaped_out_tensor_size += batch0_out_size
        out_types += [BATCH0]
        out_perm += batch0_out_positions

    if(len(convolution_out_positions) > 0): # convolution
        reshaped_out_tensor_size += convolution_out_size
        out_types += [CONVOLUTION]
        out_perm += convolution_out_positions


    return input0_perm, input1_perm, out_perm, output_subscript_conv_appended, preatomic0_tensor_size, preatomic1_tensor_size, reshaped_out_tensor_size


def conv_einsum_pair(*operands, max_mode_sizes=None, padding_mode='max_zeros', padding=0, \
                     stride=1, dilation=1, bias=False):
    
    input_subscripts, output_subscript, convolution_subscript, indices_set, operands \
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

   
    # we compute the data necessary for permuting/reshaping the tensors into, and from for the output, atomic form 
    
    nonoutput_convolution_indices_set = set(convolution_subscript) - output_subscript_set
    nonoutput_convolution0_positions = occurrence_indices(input_subscripts0, nonoutput_convolution_indices_set) 
    output_subscript_conv_appended = output_subscript + ''.join(elements(input_subscripts0, nonoutput_convolution0_positions)) 

    def convert_dict_to_tuple(parameter, output_subscript, convolution_subscript, default_value):
        # the tuple is in the same order as appearance as in the output subcript
        
        if isinstance(parameter, Mapping):
            # if the convolution index does not appear in the map, then that tuple element gets the default_value
            key_list = list(parameter.keys())
            out = [] 
            conv_occurrences = occurrence_indices(output_subscript, convolution_subscript)

            for c in conv_occurrences: 
                if output_subscript[c] in key_list:
                    out += [parameter[output_subscript[c]]]
                else:
                    out += [default_value]
            return tuple(out) 

        elif isinstance(parameter, int): 
            return tuple([parameter] * len(convolution_subscript))
        elif parameter == None:
            return tuple([0] * len(convolution_subscript))
        else:
            print("Error convert_dict_to_tuple: " + str(parameter))


    stride_tuple = convert_dict_to_tuple(stride, output_subscript_conv_appended, \
                                         convolution_subscript, 1) 
    dilation_tuple = convert_dict_to_tuple(dilation, output_subscript_conv_appended, \
                                           convolution_subscript, 1)
    padding_tuple = convert_dict_to_tuple(padding, output_subscript_conv_appended, \
                                          convolution_subscript, 0) 
    max_mode_sizes_tuple = convert_dict_to_tuple(max_mode_sizes, \
                                                 output_subscript_conv_appended, \
                                                 convolution_subscript, 0)
    
   
    # this atomic_permutation function is a monolith... Could be worth refactoring
    # currently it computes everything required for reshaping/permuting to and away from the atomic form 
    # required for convolution_atomic_operation
    # \todo atomic_permutation may have to change for max_mode_sizes....
    input0_perm, input1_perm, out_perm, output_subscript_conv_appended, \
    preatomic0_tensor_size, preatomic1_tensor_size, reshaped_out_tensor_size \
        = atomic_permutation(input_subscripts0, input_subscripts1, \
                             output_subscript_conv_appended, \
                             convolution_subscript, \
                             indices_set, \
                             summed0_tensor.size(), summed1_tensor.size(), \
                             stride_tuple, max_mode_sizes_tuple) 

   
  
    # we do the permuting and reshaping, call our atomic operation, then reshape and permute the output
    preatomic0_tensor = summed0_tensor.permute(input0_perm).reshape(preatomic0_tensor_size)
    preatomic1_tensor = summed1_tensor.permute(input1_perm).reshape(preatomic1_tensor_size)

    num_conv = len(convolution_subscript)

    unreshaped_out = convolution_atomic_operation(preatomic0_tensor, preatomic1_tensor, \
                                                  num_conv, \
                                                  max_mode_sizes=max_mode_sizes_tuple, \
                                                  padding_mode=padding_mode, \
                                                  padding=padding_tuple, \
                                                  stride=stride_tuple, \
                                                  dilation=dilation_tuple, bias=bias)

    
    reshaped_out = unreshaped_out.reshape(reshaped_out_tensor_size).permute(permutation_inverse(out_perm))

    # lastly, we must contract out any convolution indices not appearing in the output   
    # if no convolution subscripts were appended, can return immediately
    if len(output_subscript_conv_appended) == len(output_subscript):
        return reshaped_out 
    
    # \todo I think the atomic call to ConvXd can actually do this step, which would probably be faster
    contract_convolutions_str = output_subscript_conv_appended + "->" + output_subscript
    return torch.einsum(contract_convolutions_str, reshaped_out)

    



#torch_A = torch.ones(2,3)
#torch_B = torch.ones(2,2,4)
#einsum_str = "ij, ijh -> ihj | j"
#print(einsum_str + " = \n" + str(conv_einsum_pair(einsum_str, torch_A, torch_B)))


#torch_A = torch.randn(4,2)
#torch_B = torch.randn(4,20)
#einsum_str = "ij, ij -> ij | j"
#print(einsum_str + " = \n" + str(conv_einsum_pair(einsum_str, torch_A, torch_B)))
#print("ij_ij_to_ij_bar_j = \n" + str(ij_ij_to_ij_bar_j(torch_A, torch_B)))


def without_duplicates(iterable):
    # this is order preserving
    return list(OrderedDict.fromkeys(iterable))


###
# \todo I don't think it returns the correct sizes if padding_mode == max_zeros and dilation != 1
# \todo in what order does the stride tuple get mapped to the convolution indices?
# \todo write documentation for using this
# \todo Should add more warnings / usage errors for the user
#
# This function (as opposed to conv_einsum_pair) needs to be responsible for handling 
# padding_mode='max_zeros' and padding_mode='max_circular', because
# those conditions put requirements on N-ary operations (as opposed to binary)
#
#
# padding_mode : 'max_zeros', 'max_circular', 'zeros', 'circular', 'reflect', 'replicate'
# bias is only applicable if there is a single convolution
###
def conv_einsum(*variadic_operands, padding_mode='max_zeros', padding=0, \
                stride=1, dilation=1, bias=False):

    input_subscripts, output_subscript, convolution_subscript, indices_set, operands \
        = _parse_conv_einsum_input(variadic_operands) 

    #### 
    # Error handling and warnings
    # \todo add more of these
    ###

    if(len(input_subscripts) > 2):
        if not (padding_mode == 'max_zeros' or padding_mode == 'max_circular'):
            print("Warning: padding mode must be max_zeros or max_circular if input size is > 2")

        if dilation != 1 or stride != 1 or padding != 0:
            print("Warning: dilation, stride, and padding are ignored if input size is > 2")


    ###
    # No-convolution cases
    # In these cases we should resort to torch.einsum so that our tool has 
    # approximately equivalent performance with einsum in all applicable cases
    ###
    if(len(convolution_subscript) == 0):        
        return torch.einsum(*variadic_operands)
    if(len(input_subscripts) <= 1):
        # 1-way convolution is really a contraction operation (though this may not be
        # true in hyperparameter cases \todo), so convert the einsum 
        # string to one without convolution
        return torch.einsum(','.join(input_subscripts) + "->" + output_subscript, operands)

    ###
    # Binary case
    ###
    if(len(input_subscripts) == 2): 

        pair_str = input_subscripts[0] + ", " + input_subscripts[1] + " -> " \
                                       + output_subscript + " | " + convolution_subscript

        return conv_einsum_pair(pair_str, operands[0], operands[1], \
                                max_mode_sizes=None, \
                                padding_mode=padding_mode, padding=padding, \
                                stride=stride, dilation=dilation, bias=bias)


    ###
    # N-ary case, N > 2
    ###

    # \todo Note that, because in the 2-ary did we decided on the convention that the first input represents the "input" or
    #       "image" tensor and the second represents the "kernel" tensor, we were able to implement the hyperparameters "dilation",
    #       "bias", and "padding" (arbitrary padding). Because in the N-ary case there's apparently not a useful convention for enforcing
    #       these, "dilation", "bias", and arbitrary designations of paddings are not supported. 
    #       It's likely that "stride" is applicable to the N-ary case, however. Additionally, there may be special cases of conv_einsums
    #       for N > 2 for which arbitrary paddings, or bias, or dilation do apply. 
    
   
    # We handle the options "max_zeros" and "max_circular" here, because they aren't naturally handled within conv_einsum_pair,
    # as they are a condition on an N-ary operation.

    # \todo rename max_mode_sizes
    max_mode_sizes = dict()
    for index in set(convolution_subscript):
        max_mode_size_index = 0 
        for j in range(0, len(input_subscripts)):
            index_pos = input_subscripts[j].find(index)
            if index_pos != -1 and operands[j].size(index_pos) > max_mode_size_index:
                max_mode_size_index = operands[j].size(index_pos)
        max_mode_sizes[index] = max_mode_size_index
 
    print("max_mode_sizes = " + str(max_mode_sizes)) 
    
    # \todo swap this out for the pairwise sequencer
    # this simple pairwise reduction evaluates from left to right, and at each pair
    # it sets the output to be those indices in the two pair inputs which remain in 
    # any input to the right, or in the final output, and it sets
    # the convolution subscript to be those indices appearing in the total convolution subscript
    # The pairwise sequencer should get called here, and the pairwise sequence should
    # get iterated over here (it's currently implementing the basic form of iterating over
    # a pairwise sequence, it's just using the simplest possible pairwise sequence)
    
    out = operands[0] # I'm pretty sure out is a reference, and this = doesn't do a copy
    left_subscript = input_subscripts[0]
    remaining_input_subscripts = input_subscripts.copy()
    remaining_input_subscripts.pop(0)

    for i in range(1, len(input_subscripts)):
        right_subscript = input_subscripts[i]
        remaining_input_subscripts.pop(0)
        remaining_input_subscript_indices = set().union(*remaining_input_subscripts)
        

        pair_output_subscript = []
        for s in output_subscript:
            if s in left_subscript or s in right_subscript:
                pair_output_subscript.append(s)

        left_right_without_duplicates = without_duplicates(left_subscript + right_subscript)
        
        for s in left_right_without_duplicates: 
            if s in remaining_input_subscript_indices and s not in pair_output_subscript: 
                pair_output_subscript.append(s)
        
        pair_output_subscript = ''.join(pair_output_subscript)

        pair_str = left_subscript + ", " + right_subscript + " -> " \
                                  + pair_output_subscript + " | " + convolution_subscript

        # I think it might be better to parse the pair convolution_subscript, and not
        # pass the total convolution subscript, but this is convenient for now
        left_subscript = pair_output_subscript

        
        if padding_mode == 'max_zeros': 
            #print("temp")
            # \todo
            #padding 
            #out = conv_einsum_pair(pair_str, out, operands[i], \
            #                       max_mode_sizes=max_mode_sizes, padding_mode='zeros')
            temp = 5
        elif padding_mode == 'max_circular':
            # in this case 

            out = conv_einsum_pair(pair_str, out, operands[i], \
                                   max_mode_sizes=max_mode_sizes, \
                                   padding_mode='circular')

        else:
            print("padding_mode must be max_linear or max_circular in the N-ary case")

    return out




class ConvEinsumFunction(torch.autograd.Function):
    def forward(ctx, *variadic_operands, padding_mode='max_zeros', \
                padding=0, stride=1, dilation=1, bias=False):
        # \todo implement
        output = conv_einsum(variadic_operands, padding_mode, padding, stride, dilation, bias)

        return output

    def backward(ctx, grad_output):
        # \todo implement
        return torch.tensor([0])



### padding test
## Using this try to experimentally verify what the proper paddings are
#stride = 1
#dilation = 3
#for i in range(1, 12):
#    A = torch.rand(i, dtype=torch.double)
#    for j in range(1, 12):
#        B = torch.rand(j, dtype=torch.double)
#
#        last_pad = 60
#        for pad in range(0, last_pad):
#            #print(pad)
#            try:
#                AB = conv_einsum("i, i -> i | i", A, B, padding_mode='circular', padding=pad, \
#                                                        stride=stride, dilation=dilation) 
#                print(str(i) + ", " + str(j) + ": " + str(pad) + ", stride = " + str(stride) \
#                      + ", dilation = " + str(dilation))
#                break;
#            except RuntimeError:
#                if pad == last_pad-1:
#                    print(str(i) + ", " + str(j) + ": no pad")
#




#torch_A = torch.ones(2,2,2,2)
#torch_B = torch.ones(2,2)
#print(conv_einsum_pair("ijkl,ik->il|ik", torch_A, torch_B))
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






#torch_A = torch.ones(2,3,2,2,2)
#torch_B = torch.ones(2,2)
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#einsum_str = "ifjkl,ik->lif|ik"
#print("einsum_str = " + einsum_str)
#out = conv_einsum(einsum_str, torch_A, torch_B, stride={"i":1, "k":1})
#print(out)
#print(out.size())


#I = 5
#J = 7
#K = 3
#L = 4
#M = 3
#N = 2
#torch_A = torch.ones(I, J, L, M, N)
#torch_B = torch.ones(I, J, L, M, N)
#einsum_str = "ijlmn, ijlmn -> lmn | lmn"
#print(einsum_str)
##print(str(conv_einsum(einsum_str, torch_A, torch_B, padding_mode='zeros', padding={"l":1, "m":1, "n":1}, {"l":1, "m":1, "n":1})))
#print(str(conv_einsum(einsum_str, torch_A, torch_B, {"l":2, "m":1, "n":1})))




#torch_A = torch.ones(4)
#torch_B = torch.ones(5)
#torch_C = torch.ones(6)
#max_mode_size = 6
#
#einsum_str = "i,i,i -> i | i"
#out = conv_einsum(einsum_str, torch_A, torch_B, torch_C, padding_mode='max_zeros')
#print("ABC = " + str(out))
#




#torch_A = torch.ones(4)
#torch_B = torch.ones(5)
#torch_C = torch.ones(6)
#einsum_str = "i, i -> i | i"
#
#padding1 = max_zeros_padding_1d(torch_A.size(0), torch_B.size(0), max_mode_size)
#out1 = conv_einsum(einsum_str, torch_A, torch_B, padding_mode='zeros', padding=padding1)
#
#padding2 = max_zeros_padding_1d(out1.size(0), torch_C.size(0), max_mode_size)
#out2 = conv_einsum(einsum_str, out1, torch_C, padding_mode='zeros', padding=padding2)
#
#print("AB = " + str(out1))
#print("(AB)C = " + str(out2))


#padding = max_zeros_padding_1d(10,10)
#print("padding = " + str(padding))




#for i in range(1, 10):
#    for j in range(1, 20):
#        
#        torch_A = torch.ones(i)
#        torch_B = torch.ones(j)
#        einsum_str = "i, i -> i | i"
#        dilation = 5
#        stride = 8
#        padding = max_zeros_padding_1d(torch_A.size(0), torch_B.size(0), dilation=dilation, \
#                                       stride=stride)
#        out = conv_einsum(einsum_str, torch_A, torch_B, padding=padding, padding_mode='zeros', \
#                          dilation=dilation, stride=stride)
#        expected_size = max(i*dilation, j)/stride # not sure about dilation factor
#        if out.size(0) == math.floor(expected_size):
#            print(str(i) + ", " + str(j) + " " + str(out) + str(out.size()) + " " + str(expected_size))

#ker = torch.rand(5, dtype=torch.double) 
#img = torch.rand(100, dtype=torch.double)
#max_mode_size = 105
#max_padding = max_zeros_padding_1d(ker.size(0), img.size(0), max_mode_size=max_mode_size)
#print("max_zeros_padding_1d = " + str(max_padding))
#outAB = conv_einsum_pair("i, i -> i | i", ker, img, max_mode_size=max_mode_size, padding_mode='zeros', padding=max_padding)
##outBA = conv_einsum_pair("i, i -> i | i", img, ker, max_mode_size=max_mode_size, padding_mode='zeros', padding=max_zeros_padding_1d(img.size(0), ker.size(0), max_mode_size=max_mode_size))
##print(outAB - outBA)
#print(outAB.size())
##print(outBA.size())
        

#### associativity test


#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(50, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
##padding = 4 # zeros for 5 x 5 -> 5
#padding = 50
##padding = 25
##AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='zeros', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#
#padding = 4 # zeros 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='zeros', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # zeros
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='zeros', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # zeros
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='zeros', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)
#
#




#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(50, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
##padding = 4 # circular for 5 x 5 -> 5
#padding = 50
##padding = 25
##AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#
#padding = 4 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)
#
        

#op12.size() = torch.Size([4, 3, 224, 224])
#self.layer_nodes['core'].size() = torch.Size([7, 3, 7, 7])
#einsum_str = "imkl, jnkl -> ijkl | kl"
#einsum_str = "jnkl, imkl -> ijkl | kl"
#print(einsum_str)
#A = torch.ones(4,3,224,224)
#B = torch.ones(7,3,7,7)
#out = conv_einsum(einsum_str, B, A, stride={"k":2, "l":2}, padding_mode='zeros', padding={"k":3, "l":3}) 
#print("out.size = " + str(out.size()))


#torch_A = torch.ones(20)
#torch_B = torch.ones(30)
#einsum_str = "i, i -> i | i"
#print(einsum_str + " = \n" + str(conv_einsum(einsum_str, torch_A, torch_B, stride=3)))


#I = 50 
#J = 10
#K = 15
#L = 20
#M = 25
## [30, 15, 30, 10, 25]
## [J,   K,  J,  L,  M]
#torch_A = torch.ones(L,M,K,I,J)
#torch_B = torch.ones(M,K,J,I)
#print("I = " + str(I) + " : J = " + str(J) + " : K = " + str(K) + " : L = " + str(L) + " : M = " + str(M))
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#einsum_str = "lmkij, mkji -> lijkm | mij"
#print("einsum_str = " + einsum_str)
#out = conv_einsum(einsum_str, torch_A, torch_B, stride={"i":10, "j":5})
##out = conv_einsum(einsum_str, torch_A, torch_B)
#print(out.size())

#torch_A = torch.ones(4)
#torch_B = torch.ones(4,5)
#torch_C = torch.ones(4,5,6)
#einsum_str = "i,ij,ijk -> ij | ij"
#print(einsum_str + " = \n" + str(conv_einsum(einsum_str, torch_A, torch_B, torch_C)))



#K = 5
#I = 5
#kernel = torch.rand(1,1,1,K)
#input_tens = torch.rand(1,1,1,I)
#
#stride = 1
#dilation = 1
#padding = 11
#padding_mode = 'circular'
#
#batch_size = input_tens.size(0)
#kernel_size = kernel.size(3)
#input_size = input_tens.size(3) # image_size is perhaps a better name
#conv_len = input_size
#groups = kernel.size(0)
#in_channels = groups*kernel.size(2)
#out_channels = groups*kernel.size(1)
#
#m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, \
#                    kernel_size=kernel_size, \
#                    stride=stride, padding=padding, padding_mode=padding_mode, \
#                    dilation=dilation, \
#                    groups=groups)
#output = m(input_tens.reshape(1, 1, input_size))
#print("m = " + str(m))
#print("output.size() = " + str(output.size()))
##print("max_circular = " + str(max_circular_padding_1d(kernel_size, input_size)))
#print("padding = " + str(padding))




#
#
#
#####################################################################
#A = torch.rand(50, dtype=torch.double)
#B = torch.rand(50, dtype=torch.double)
#C = torch.rand(60, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#padding = 59 # circular
##padding = 25
#AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
##AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#
#padding = 59 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 49 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 49 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)
#
#####################################################################
# 




#####################################################################
### associativity test zeros same size
#
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(5, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#
#padding = max_zeros_padding_1d(5, 5, max_mode_sizes["i"])
##AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='zeros', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#
#
#padding = max_zeros_padding_1d(5, 5, max_mode_sizes["i"])
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='zeros', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#
#
#padding = max_zeros_padding_1d(5, 5, max_mode_sizes["i"])
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='zeros', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#
#padding = max_zeros_padding_1d(5, 5, max_mode_sizes["i"])
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='zeros', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)
#



        

######################################################################
#### associativity test circular same size
#
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(5, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#padding = 4 # circular
##padding = 25
##AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#
#padding = 4 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)
#
#
#
####################################################################


##### Binary zeros

#I1 = 8
#I2 = 8
#A = torch.ones(I1, dtype=torch.double)
#B = torch.ones(I2, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0))
##max_mode_sizes = {"i": max_mode_size}
#max_mode_sizes = {"i": 100}
#
#
#padding = max_zeros_padding_1d(8, 8, max_mode_sizes["i"])
#AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
##AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='zeros', padding=padding)
#print("\nAB.size() = " + str(AB.size()))
#print("AB = " + str(AB) + "\n")
#





####### Binary circular 
#A = torch.ones(5, dtype=torch.double)
#B = torch.ones(8, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0) )
#max_mode_sizes = {"i": max_mode_size}
##max_mode_sizes = {"i": 12}
#padding = 4 
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("\nAB.size() = " + str(AB.size()))
#print("AB = " + str(AB) + "\n")
#



#######################################################################
##### associativity test with full + circular convolutinos
#### The idea is for the A*B to do a full convolution, and then finish the (AB)*C with a padded
#### circular
#### For the B*C do a padded circular, and then finish the A*(BC) with a padded circular
#
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(50, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#padding = 4 # circular
##padding = 25
#AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes={"i": 9}, \
##AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='zeros', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
## this is a full convolution since padding = ker_size - 1 and padding_mode='zeros',
## The output size should be o = i + k - 1 = 9
#
#padding = 8 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
#
#print(AB_C - A_BC)
######################################################################
#



##########
########## This test shows that the naive max padding binary circular convolution is not associative 
##########
#
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(9, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#padding = 4 
##padding = 25
##AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes={"i": 9}, \
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("AB = " + str(AB))
#print("AB.size() = " + str(AB.size()) + "\n\n")
## this is a full convolution since padding = ker_size - 1 and padding_mode='zeros',
## The output size should be o = i + k - 1 = 9
#
#padding = 4 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
#
#
#print("AB_C = " + str(AB_C))
#print("A_BC = " + str(A_BC))
#



#### test case(2), 1
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(6, dtype=torch.double)
#
## since 6 < 5 + 5 - 1, case (2), so we pad the image tensor on the left to 6 with 0s and then 
## do the circular max padding with pytorch
#
#B_zero_pad = torch.cat([torch.zeros(1, dtype=torch.double), B]) 
#AB = conv_einsum_pair("i, i -> i | i", A, B_zero_pad, padding_mode='circular', padding=4)
#
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, padding_mode='circular', padding=5)
#
#BC = conv_einsum_pair("i, i -> i | i", B, C, padding_mode='circular', padding=5)
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, padding_mode='circular', padding=5)
#
#print("A.size() = " + str(A.size()))
#print("B.size() = " + str(B.size()))
#print("C.size() = " + str(C.size()))
#print("AB.size() = " + str(AB.size()))
#print("BC.size() = " + str(BC.size()))
#print("AB_C = " + str(AB_C))
#print("A_BC = " + str(A_BC))


#### test case(2), 1b: checking if it doesn't matter if kernel or image is padded
####                   It does appear to matter
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(6, dtype=torch.double)
#
## since 6 < 5 + 5 - 1, case (2), so we pad the image tensor on the left to 6 with 0s and then 
## do the circular max padding with pytorch
#
#A_zero_pad = torch.cat([torch.zeros(1, dtype=torch.double), A]) 
#AB = conv_einsum_pair("i, i -> i | i", A_zero_pad, B, padding_mode='circular', padding=6)
#
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, padding_mode='circular', padding=5)
#
#BC = conv_einsum_pair("i, i -> i | i", B, C, padding_mode='circular', padding=5)
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, padding_mode='circular', padding=5)
#
#print("A.size() = " + str(A.size()))
#print("B.size() = " + str(B.size()))
#print("C.size() = " + str(C.size()))
#print("AB.size() = " + str(AB.size()))
#print("BC.size() = " + str(BC.size()))
#print("AB_C = " + str(AB_C))
#print("A_BC = " + str(A_BC))


#### test case(2), 1
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(9, dtype=torch.double)
#
#B_zero_pad = torch.cat([torch.zeros(4, dtype=torch.double), B]) 
#AB_circular = conv_einsum_pair("i, i -> i | i", A, B_zero_pad, padding_mode='circular', padding=8)
#AB_C_circular = conv_einsum_pair("i, i -> i | i", AB_circular, C, padding_mode='circular', padding=8)
#
#
#AB_full = conv_einsum_pair("i, i -> i | i", A, B, padding_mode='zeros', padding=4, max_mode_sizes={"i":9}) 
#AB_C_full = conv_einsum_pair("i, i -> i | i", AB_full, C, padding_mode='circular', padding=8)
#
#BC = conv_einsum_pair("i, i -> i | i", B, C, padding_mode='circular', padding=8)
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, padding_mode='circular', padding=8)
#
#print("A.size() = " + str(A.size()))
#print("B.size() = " + str(B.size()))
#print("C.size() = " + str(C.size()))
#print("AB_full.size() = " + str(AB_full.size()))
#print("AB_full = " + str(AB_full))
#print("AB_circular.size() = " + str(AB_circular.size()))
#print("AB_circular = " + str(AB_circular))
#print("BC.size() = " + str(BC.size()))
#print("AB_C_circular = " + str(AB_C_circular))
#print("AB_C_full = " + str(AB_C_full))
#print("A_BC = " + str(A_BC))

####
# I think the cases for deciding which kind of circular convolution to do can be
# decided in the pairwise reduction 
# Basically we have the max_mode_sizes dictionary...







A = torch.rand(5, dtype=torch.double)
B = torch.rand(5, dtype=torch.double)
C = torch.rand(20, dtype=torch.double)

# A: [-2, 2]
# B: [-2, 2]
# C: [-9.5, 9.5]

B_zero_pad_left = torch.cat([torch.zeros(15, dtype=torch.double), B]) 
AB_circular = conv_einsum_pair("i, i -> i | i", A, B_zero_pad_left, padding_mode='circular', padding=4)
print("AB_circular.size() = " + str(AB_circular.size()))
print("AB_circular = " + str(AB_circular))
# B_left: [-17, 2], padding: [?, ?] 
# AB_circular: [-17, 2] 

AB_C_circular = conv_einsum_pair("i, i -> i | i", AB_circular, C, padding_mode='circular', padding=19)
# AB_C_circular: [-17, 2] * [-9.5, 9.5] = [-17, 2] -> -7.5

B_zero_pad_right = torch.cat([B, torch.zeros(15, dtype=torch.double)]) 
AB_circular_right = conv_einsum_pair("i, i -> i | i", A, B_zero_pad_right, padding_mode='circular', padding=4)
AB_C_circular_right = conv_einsum_pair("i, i -> i | i", AB_circular_right, C, padding_mode='circular', padding=19)
# B_right: [-2, 17]
# AB_circular_right: [-2, 17] 
# AB_C_circular_right: [-2, 17] * [-9.5, 9.5] = [-2, 17] -> 7.5

B_zero_pad_mid = torch.cat([torch.zeros(7, dtype=torch.double), B, torch.zeros(8, dtype=torch.double)]) 
AB_circular_mid = conv_einsum_pair("i, i -> i | i", A, B_zero_pad_mid, padding_mode='circular', padding=4)
AB_C_circular_mid = conv_einsum_pair("i, i -> i | i", AB_circular_mid, C, padding_mode='circular', padding=19)
# B_mid: [-9, 10]
# AB_circular_mid: [-9, 10]
# AB_C_circular_mid = [-9, 10] * [-9.5, 9.5] = [-9, 10] -> .5


B_zero_pad_one = torch.cat([torch.zeros(14, dtype=torch.double), B, torch.zeros(1, dtype=torch.double)]) 
AB_circular_one = conv_einsum_pair("i, i -> i | i", A, B_zero_pad_one, padding_mode='circular', padding=4)
AB_C_circular_one = conv_einsum_pair("i, i -> i | i", AB_circular_one, C, padding_mode='circular', padding=19)
# B_one: [-16, 3]
# AB_circular_one:  [-16, 3]
# AB_C_circular_one: [-16, 3] * [-9.5, 9.5] = [-16, 3] -> -6.5



AB_full = conv_einsum_pair("i, i -> i | i", A, B, padding_mode='zeros', padding=4, max_mode_sizes={"i":9}) 
AB_C_full = conv_einsum_pair("i, i -> i | i", AB_full, C, padding_mode='circular', padding=8)
# AB_full: [-2, 2] *_full [-2, 2] = [-4, 4]
# AB_C_full: [-4, 4] *_circular [-9.5, 9.5] = [-9.5, 9.5] -> 0

BC = conv_einsum_pair("i, i -> i | i", B, C, padding_mode='circular', padding=4)
A_BC = conv_einsum_pair("i, i -> i | i", A, BC, padding_mode='circular', padding=19)

AB = conv_einsum_pair("i, i -> i | i", A, B, padding_mode='circular', padding=4)
# 
AB_C = conv_einsum_pair("i, i -> i | i", AB, C, padding_mode='circular', padding=19)


print("A.size() = " + str(A.size()))
print("B.size() = " + str(B.size()))
print("C.size() = " + str(C.size()))
#print("\n\n")
#print("AB_full.size() = " + str(AB_full.size()))
#print("AB_full = " + str(AB_full))
#print("AB_circular.size() = " + str(AB_circular.size()))
#print("AB_circular = " + str(AB_circular))
#print("AB_circular_right.size() = " + str(AB_circular_right.size()))
#print("AB_circular_right = " + str(AB_circular_right))
#print("AB_circular_mid.size() = " + str(AB_circular_mid.size()))
#print("AB_circular_mid = " + str(AB_circular_mid))
print("\n\n")
print("AB_C_circular_left = " + str(AB_C_circular))
print("AB_C_circular_right = " + str(AB_C_circular_right))
print("AB_C_circular_mid = " + str(AB_C_circular_mid))
print("AB_C_circular_one = " + str(AB_C_circular_one))
print("\n\n")
print("AB_C_full = " + str(AB_C_full))
print("\n\n")

# if you always convolve with the tensors of highest length it should work,
# because the max circular padding seems to be properly supported by pytorch.
#print("A_BC = " + str(A_BC))

# but this one should be wrong
#print("(Note the error) AB_C = " + str(AB_C))

