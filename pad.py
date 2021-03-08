'''



'''

import math


def padding_mode_zeros_L_out(kernel_size, input_size, padding, stride,
                             dilation):
    '''
    Calculate the expected output length for a given 1d convolution with
    padding mode 'zeros'.

    Reference https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    '''

    return math.floor(
        input_size + (2*padding - dilation*(kernel_size-1) - 1) / stride + 1
        )


def max_zeros_padding_1d(ker_mode_size, input_mode_size, max_mode_size,
                         stride=1, dilation=1):
    '''
    This computes the zero padding to append to the input image vector so that
    the resulting convolution has the same length as the maximum length of the
    kernel and the input tensor. Note the kernel can be longer than the input
    tensor.

    Additionally,
    conv_einsum("i, i -> i | i", A, B, padding=)
    conv_einsum("i, i -> i | i", B, A, padding=)

    when this padding is selecting. These two properties may not hold when
    dilation > 1

    Reference
    https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
    Section shape
    We add 1 so the integer division by 2 errs large instead of small

    '''

    # TODO fix problem mentioned in original document

    max_ker_input = max(ker_mode_size, input_mode_size, max_mode_size)
    twice_padding = ((max_ker_input-1) * stride + 1
                     + dilation*(ker_mode_size-1) - input_mode_size)
    return (twice_padding+1)//2


def max_zeros_padding_nd(ker_mode_size, input_mode_size,
                         max_mode_size, stride=1, dilation=1):
    '''


    '''

    return tuple(max_zeros_padding_1d(ker_mode_size[i], input_mode_size[i],
                 max_mode_size[i], stride[i], dilation[i])
                 for i in range(0, len(ker_mode_size)))
