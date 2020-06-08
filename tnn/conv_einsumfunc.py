
import torch
import torch.utils.dlpack



from parse_conv_einsum import _parse_conv_einsum_input
#import parse_conv_einsum

#import sys
#import os
# \todo need to figure out where to put the lib/tnn.so binary so that it adheres to
#       usual import semantics
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bin'))
#import tnnlib



# expects pytorch tensors for now
def cpp_conv_einsum(*operands):
    input_subscripts, output_subscript, convolution_subscript, subscripts_set, operands \
        = _parse_conv_einsum_input(operands)

    dlpack_operands = [] 
    for operand in operands:
        dlpack_operands.append(torch.utils.dlpack.to_dlpack(operand))


    dlpack_out = tnnlib.conv_einsum(input_subscripts, output_subscript, convolution_subscript, \
                                    subscripts_set, dlpack_operands)



