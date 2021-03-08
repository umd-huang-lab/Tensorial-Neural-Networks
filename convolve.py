'''



'''

from pad import max_zeros_padding_nd
import torch


def calc_conv1d(kernel, input_tens, max_mode_size=None, padding_mode='zeros',
                padding=0, stride=1, dilation=1, bias=False):

    '''
    calculate "ijkl, mikl -> mijl | l"

    this is supposed to compute the most general, up to reshaping / permuting,
    convolutional einsum with 1 convolution index which is computable by Conv1d
    alone. This is the order the indices must appear in so that the operation
    can be done without any calls to permute.

    input = (batch_size = M, input_channel = groups*K = I*K,
             input_size = input_tens.size(3))
    weight = (out_channels = groups * J = I * J, in_channels/groups = K,
              kernel_size = kernel.size(3))
    output = (batch_size = M, out_channels = groups * J = I * J,
              conv_len = max(input_size, kernel_size))

    and it appears the indices are laid out as (so, after reshaping)

    input: M, I, K, L
    weight: I, J, K, L
    output: M, I, J, L

    [Reference] https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html



    '''

    batch_size = input_tens.size(0)
    kernel_size = kernel.size(3)

    input_size = input_tens.size(3)  # image_size is perhaps a better name
    conv_len = max(kernel_size, input_size)
    if max_mode_size is not None:
        conv_len = max(conv_len, max_mode_size)

    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)

    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, padding_mode=padding_mode,
                        dilation=dilation, groups=groups, bias=bias)

    kernel_flipped = torch.flip(kernel, [3])

    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups,
                                        kernel_size)

    output = m(input_tens.reshape(batch_size, in_channels, input_size))
    out_shape = (batch_size, groups, out_channels//groups, conv_len//stride)

    # \todo Cutting the mode down only makes sense for a max padding
    #       or if they're passing the maximum expected size
    return torch.reshape(output.data[:, :, :(conv_len//stride)], out_shape)


def calc_conv2d(kernel, input_tens, max_mode_sizes=None, padding_mode='zeros',
                padding=0, stride=1, dilation=1, bias=False):

    '''

    calculate "ijklm, niklm -> nijlm|lm"


    '''

    batch_size = input_tens.size(0)
    kernel_size = kernel.size()[3:5]
    input_size = input_tens.size()[3:5]

    max_h = max(kernel_size[0], input_size[0])
    max_w = max(kernel_size[1], input_size[1])
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)

    m = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, padding_mode=padding_mode,
                        dilation=dilation, groups=groups, bias=bias)

    kernel_flipped = torch.flip(kernel, [3, 4])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups,
                                        kernel_size[0], kernel_size[1])

    output = m(input_tens.reshape(batch_size, in_channels, input_size[0],
               input_size[1]))

    try:
        stride[0]
    except TypeError:
        stride = [stride]*2

    out_shape = (batch_size, groups, out_channels//groups, max_h//stride[0],
                 max_w//stride[1])

    return torch.reshape(output.data[:, :,
                                     :(max_h//stride[0]), :(max_w//stride[1])],
                         out_shape)


def calc_conv3d(kernel, input_tens, max_mode_sizes=0, padding_mode='zeros',
                padding=0, stride=1, dilation=1, bias=False):

    '''
    calculate "ijklmn, oiklmn -> oijlmn | lmn"



    '''

    batch_size = input_tens.size(0)
    kernel_size = kernel.size()[3:6]
    input_size = input_tens.size()[3:6]
    max_d = max(kernel_size[0], input_size[0])
    max_h = max(kernel_size[1], input_size[1])
    max_w = max(kernel_size[2], input_size[2])
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)

    m = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, padding_mode=padding_mode,
                        dilation=dilation, groups=groups, bias=bias)

    kernel_flipped = torch.flip(kernel, [3, 4, 5])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups,
                                        kernel_size[0], kernel_size[1],
                                        kernel_size[2])

    output = m(input_tens.reshape(batch_size, in_channels, input_size[0],
               input_size[1], input_size[2]))

    try:
        stride[0]
    except TypeError:
        stride = [stride]*3

    out_shape = (batch_size, groups, out_channels//groups, max_d//stride[0],
                 max_h//stride[1], max_w//stride[2])
    return torch.reshape(output.data[:, :,
                                     :(max_d//stride[0]), :(max_h//stride[1]),
                                     :(max_w//stride[2])],
                         out_shape)


def convolution_atomic_operation(kernel, input_tens, num_convolutions,
                                 max_mode_sizes, padding_mode='max_zeros',
                                 padding=0, stride=1, dilation=1, bias=False):

    '''
    This operation expects the inputs input_tens and kernel to be shaped
    and permuted according to the "atomic forms" given by the following
    functions note the convolution indices always appear at the end


    '''

    if padding_mode == 'max_zeros':
        kernel_size = kernel.size()[3:len(kernel.size())]
        input_size = input_tens.size()[3:len(input_tens.size())]
        padding = max_zeros_padding_nd(kernel_size, input_size,
                                       max_mode_size=max_mode_sizes,
                                       stride=stride, dilation=dilation)

        padding_mode = 'zeros'

    if num_convolutions == 0:
        print("Error: Expects at least one convolution index")

    elif num_convolutions == 1:
        stride = stride[0]
        dilation = dilation[0]
        padding = padding[0]
        max_mode_size = max_mode_sizes[0]

        return calc_conv1d(kernel, input_tens, max_mode_size=max_mode_size,
                           padding_mode=padding_mode, padding=padding,
                           stride=stride, dilation=dilation, bias=bias)

    elif num_convolutions == 2:
        return calc_conv2d(kernel, input_tens, max_mode_sizes=max_mode_sizes,
                           padding_mode=padding_mode, padding=padding,
                           stride=stride, dilation=dilation, bias=bias)

    elif num_convolutions == 3:
        return calc_conv3d(kernel, input_tens, max_mode_sizes=max_mode_sizes,
                           padding_mode=padding_mode, padding=padding,
                           stride=stride, dilation=dilation, bias=bias)
    else:
        print("convolution_atomic_operation num_convolutions >= 4 \
              not implemented")
