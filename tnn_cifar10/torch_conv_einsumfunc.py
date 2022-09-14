
import torch
import torch.nn.functional as F

# import the checkpoint API
import torch.utils.checkpoint as checkpoint

#from .parse_conv_einsum import _parse_conv_einsum_input
from parse_conv_einsum import _parse_conv_einsum_input

from collections import OrderedDict
from collections.abc import Mapping

from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear

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
#
def max_zeros_padding_1d(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):
    # see https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html for the definition
    # under the Shape section for the definition of the padding. We add 1 so the integer division
    # by 2 errs large instead of small
    # for 1d convolutions ker_mode_size < input_mode_size but this is not true for higher
    # dimensions, and we use this function to compute those paddings

    # "input" tensor is synonymous with "image" tensor in much of this

    max_ker_input = max(ker_mode_size, input_mode_size, max_mode_size)


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


def max_circular_padding_1d(ker_mode_size, input_mode_size, \
                            stride=1, dilation=1):


    # the formulas in this function were determined by experimentally computing the first pad
    # which did not cause a run time error
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


    kernel_flipped = torch.flip(kernel, [3])

    ins = input_tens.reshape(batch_size, in_channels, input_size)
    filts = kernel_flipped.view(out_channels, in_channels//groups, kernel_size)
    output = F.conv1d(ins, filts, padding=padding,stride=stride, bias=bias,dilation=dilation, groups=groups)

    out_shape = (batch_size, groups, out_channels//groups, conv_len//stride)

    return torch.reshape(output[:,:,:(conv_len//stride)], out_shape)


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


    kernel_flipped = torch.flip(kernel, [3,4])
    ins = input_tens.reshape(batch_size, in_channels, input_size[0], input_size[1])
    filts = kernel_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])
    
    output = F.conv2d(ins, filts, padding=padding,stride=stride, bias=bias,dilation=dilation, groups=groups)
    
    
    try:
        stride[0]
    except TypeError:
        stride = [stride]*2
    
    out_shape = (batch_size, groups, out_channels//groups, max_h//stride[0], max_w//stride[1])
    final_out = torch.reshape(output[:,:,:(max_h//stride[0]),:(max_w//stride[1])], out_shape)
    return final_out



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
                        stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation, groups=groups, bias=True)

    kernel_flipped = torch.flip(kernel, [3,4,5])
    
    
    ins = input_tens.reshape(batch_size, in_channels, input_size[0], input_size[1], input_size[2])
    filts = kernel_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2])
    
    output = F.conv3d(ins, filts, padding=padding,stride=stride, bias=bias,dilation=dilation, groups=groups)
    
    
    try:
        stride[0]
    except TypeError:
        stride = [stride]*3
    
    out_shape = (batch_size, groups, out_channels//groups, max_d//stride[0], max_h//stride[1], max_w//stride[2])

    return torch.reshape(output[:,:,:(max_d//stride[0]),:(max_h//stride[1]),:(max_w//stride[2])], out_shape)


#
def convolution_atomic_operation(kernel, input_tens, num_convolutions, max_mode_sizes, \
                                 padding_mode='max_zeros', padding=0, stride=1, dilation=1, \
                                 bias=True):
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
        def custom(A,B):
            return contract(standard_einsum_str, A, B)
        return checkpoint(custom, operands[0], operands[1])
    #return contract(standard_einsum_str, operands[0], operands[1])


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
        summed0_tensor = contract(appear_once0_einsum_str, operands[0])
    else:
        summed0_tensor = operands[0]
    if(len(non_outputs_appear_once1) > 0):
        summed1_tensor = contract(appear_once1_einsum_str, operands[1])
    else:
        summed1_tensor = operands[1]

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
    def custom(A):
        return contract(contract_convolutions_str,A)
    return checkpoint.checkpoint(custom,reshaped_out)
#return contract(contract_convolutions_str,reshaped_out)





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
        return contract(*variadic_operands)

    if(len(input_subscripts) <= 1):
        # 1-way convolution is really a contraction operation (though this may not be
        # true in hyperparameter cases \todo), so convert the einsum
        # string to one without convolution
        def custom(inputs):
            return contract(','.join(input_subscripts) + "->" + output_subscript, inputs)
#        return checkpoint.checkpoint(contract, inputs)
        return contract(','.join(input_subscripts) + "->" + output_subscript, inputs)

    ###
    # Binary case
    ###



    if(len(input_subscripts) == 2):

        pair_str = input_subscripts[0] + ", " + input_subscripts[1] + " -> " \
                                       + output_subscript + " | " + convolution_subscript

#        def custom1(A,B):
#            return conv_einsum_pair(pair_str, A, B, \
#                                  max_mode_sizes=None, \
#                                  padding_mode=padding_mode, padding=padding, \
#                                  stride=stride, dilation=dilation, bias=bias)

        #out = checkpoint.checkpoint(custom, operands[0], operands[1])
#        print("inside input:", operands[1].shape,"inside kernel:", operands[0].shape, "inside outside",out.shape,\
#              "padding mode:", padding_mode, "padding:", padding, "stride:", stride, "dilation:",dilation, "bias:",bias,"\n")
        return conv_einsum_pair(pair_str, operands[0], operands[1], \
                                max_mode_sizes=None, \
                                padding_mode=padding_mode, padding=padding, \
                                stride=stride, dilation=dilation, bias=bias)
#return checkpoint.checkpoint(custom1,operands[0], operands[1])


    ###
    # N-ary case, N > 2
    ###

    #Note that, because in the 2-ary did we decided on the convention that the first input represents the "input" or
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

    # this simple pairwise reduction evaluates from left to right, and at each pair
    # it sets the output to be those indices in the two pair inputs which remain in
    # any input to the right, or in the final output, and it sets
    # the convolution subscript to be those indices appearing in the total convolution subscript
    # The pairwise sequencer should get called here, and the pairwise sequence should
    # get iterated over here (it's currently implementing the basic form of iterating over
    # a pairwise sequence, it's just using the simplest possible pairwise sequence)

    out = operands[0]
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
            def custom(A,B):
                return conv_einsum_pair(pair_str, A, B, \
                                       max_mode_sizes=max_mode_sizes, \
                                       padding_mode='circular')

#            out = checkpoint.checkpoint(custom,out,operands[i]);
            return conv_einsum_pair(pair_str, out, operands[i], \
                                    max_mode_sizes=max_mode_sizes, \
                                    padding_mode='circular')

        else:
            print("padding_mode must be max_linear or max_circular in the N-ary case")

    return out
