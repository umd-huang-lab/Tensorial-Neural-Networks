'''


'''


from parse_conv_einsum import _parse_conv_einsum_input
from convolve import convolution_atomic_operation
from reformat import permutation_inverse, occurrence_indices, atomic_permutation, elements

from collections import OrderedDict
from collections.abc import Mapping
import torch


def conv_einsum_pair(*operands, max_mode_sizes=None, padding_mode='max_zeros',
                     padding=0, stride=1, dilation=1, bias=False):

    '''


    '''

    input_subscripts, output_subscript, convolution_subscript, indices_set, operands \
        = _parse_conv_einsum_input(operands)

    input_subscripts0 = input_subscripts[0]
    input_subscripts1 = input_subscripts[1]

    # we remove from the convolution_subscript the convolution indices appearing
    # in only one of the two tensors, because this is handled as a contraction,
    # and additionally may be summed out if it is not an output

    nontrivial_convolution_subscript_list = []
    for c in convolution_subscript:
        if (c in input_subscripts0) and (c in input_subscripts1):
            nontrivial_convolution_subscript_list += c
    convolution_subscript = ''.join(nontrivial_convolution_subscript_list)

    # If after removing nontrivial convolution indices there are no convolution
    # indices, do the computation using Pytorch's contraction-only einsum

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


def without_duplicates(iterable):
    # this is order preserving
    return list(OrderedDict.fromkeys(iterable))


def conv_einsum(*variadic_operands, padding_mode='max_zeros', padding=0,
                stride=1, dilation=1, bias=False):

    input_subscripts, output_subscript, convolution_subscript, indices_set, \
        operands = _parse_conv_einsum_input(variadic_operands)

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
            #print("pair_str", pair_str, "out shape", out.shape, "operands[i]", operands[i].shape)
            out = conv_einsum_pair(pair_str, out, operands[i], \
                                   max_mode_sizes=max_mode_sizes, padding_mode='zeros')

        elif padding_mode == 'max_circular':
            # in this case

            out = conv_einsum_pair(pair_str, out, operands[i], \
                                   max_mode_sizes=max_mode_sizes, \
                                   padding_mode='circular')

        else:
            print("padding_mode must be max_linear or max_circular in the N-ary case")

    return out
