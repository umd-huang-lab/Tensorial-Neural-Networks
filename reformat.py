'''

reshape and permuate


'''


def occurrence_indices(collection, possible_occurrences):
    return [i for i in range(0, len(collection))
            if collection[i] in possible_occurrences]


def elements(collection, occurrence_indices_list):
    return [collection[i] for i in occurrence_indices_list]


def permutation_indices(A, permutation_of_A_subcollection):
    '''
    this returns the index form of the permutation taking the corresponding
    subcollection of A topermutation_of_A_subcollection, where A is indexed
    collection and permutation_of_A_subcollection is some ordered subcollection
    of A this function can also be used like an ordered_occurrence_indices,
    except the result is always a permutatation (so it gives the relative
    ordered of the subcollection, not the order of the subcollection in the
    bigger collection)


    '''
    return [x for x, _ in
            sorted(zip(range(0, len(permutation_of_A_subcollection)),
                   permutation_of_A_subcollection),
                   key=lambda perm_el: A.index(perm_el[1]))]


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


def atomic_permutation(input_subscripts0, input_subscripts1,
                       output_subscript_conv_appended, convolution_subscript,
                       indices_set, input0_tensor_size, input1_tensor_size,
                       stride_tuple, max_mode_sizes_tuple):

    '''
    This computes the permutations necessary to permute the two input tensors
    and hypothetical output tensor into atomic form.

    This also returns number of indices of each type: group, batch0, batch1,
    contraction, convolution

    This also returns the convolution indices which are supposed to be summed
    out after the convolution is done

    This assumes there are no nonoutput indices appearing in only one tensor.
    This also assumes any convolution index appears in both tensors.

    A convolution index appearing in only one of the two tensors
    should be turned into a contraction index.

    Due to the definitions of Conv1d, Conv2d, and Conv3d, we need to permute
    our tensors so that the indices are laid out as follows:
    (group indices, batch0 indices, contraction indices, convolution indices)
    x (batch1 indices, group indices, contraction indices, convolution indices)
     -> (batch1 indices, group indices, batch0 indices, convolution indices)

    The indices of each type must appear in the same order on both sides.
    That order is determined by the order they appear as outputs for output
    indices, and the order they appear in the first tensor for nonoutput
    indices (which are contractions and convolutions without outputs)

    Also, if a convolution index is not an output, then we must add it in as an
    output and sum it out at the end. It is added after the convolution indices
    which are outputs, in the order it appears in the first tensor.


    this can surely be optimized, it is probably not a bottleneck though

    TODO:
    there may be a cleaner way to organize this code, also I tacked on
    input0_tensor_size and input1_tensor_size and the computation those are
    used for


    '''

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

    # positions in input_subscripts0 which have group indices
    group0_positions = occurrence_indices(input_subscripts0, group_indices_set)
    # likewise for input_subscripts1
    group1_positions = occurrence_indices(input_subscripts1, group_indices_set)
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
    # If we permute the batch inputs so that they appear in the same order as
    # they do in the output, then the atomic operation computes
    # them in the order that they will appear in the final output,
    # so no additional permuting is required

    batch0_perm = permutation_indices(elements(
                                                input_subscripts0,
                                                batch0_positions),
                                      elements(output_subscript_conv_appended,
                                      batch0_out_positions))

    batch1_perm = permutation_indices(elements(input_subscripts1, batch1_positions),
                                      elements(output_subscript_conv_appended,
                                      batch1_out_positions))

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
