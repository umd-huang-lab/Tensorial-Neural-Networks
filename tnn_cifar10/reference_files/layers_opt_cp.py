import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.modules.utils as utils

from torch.utils.checkpoint import checkpoint


# utility function for factorization
def factorize(number, order):
    d = {
        (1, 3): [1, 1], (1, 4): [1, 1, 1],
        (3, 3): [1, 3], (3, 4): [1, 1, 3],
        (4, 3): [2, 2], (4, 4): [1, 2, 2],
        (8, 3): [2, 4], (8, 4): [2, 2, 2],
        (16, 3): [4, 4], (16, 4): [2, 2, 4],
        (32, 3): [4, 8], (32, 4): [2, 4, 4],
        (64, 3): [8, 8], (64, 4): [4, 4, 4],
        (128, 3): [8, 16], (128, 4): [4, 4, 8],
        (256, 3): [16, 16], (256, 4): [4, 8, 8],
        (512, 3): [16, 32], (512, 4): [8, 8, 8],
        (1024, 3): [32, 32], (1024, 4): [8, 8, 16]
    }
    return d[(number, order)]


def solve_quadratic_equation(a, b, c):
  return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2*a) \
    if abs(a) > 1e-6 else (-c / b)


## Reshaped CP Convolutional Layer
class RCP_O4_Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, rank=None, cr=None,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        """
        Initialization of reshaped CP 2D-convolutional layer.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        in_channels:  int
            Number of channels of the input tensor.
        out_channels: int
            Number of channels of the output tensor.

        (Hyper-parameters of the CP decomposition)
        kernel_size: int
            The length of the convolutional kernels.
        rank: int
            The rank of CP decomposition.
            default: 8

        (Hyper-parameters of the convolutional operation)
        stride: int
            The stride of convolutional kernels.
            Default: 1
        dilation: int
            The spacing between the elements in the convolutional kernels.
            Default: 1
        padding: int
            The number padding entries added to both side of the input.
            Default: 0
        padding_mode: str ("zeros", "reflect", "replicate" or "circular")
            The padding mode of the convolutional layer.
            Default: "circular"

        (additional hyper-parameters)
        groups: int
            The number of interconnected blocks in the convolutional layer.
            Note: Both in_channels and out_channels should be divisible by groups.
            Default: 1
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True

        """
        super(RCP_O4_Conv2d, self).__init__()

        # Input/output interface
        self.groups = groups

        self.in_channels = in_channels
        self.out_channels = out_channels

        assert self.in_channels % self.groups == 0, \
            "The input channels should be divisble by the number of groups."
        assert self.out_channels % self.groups == 0, \
            "The output channels should be divisble by number of groups."

        # CP decomposition
        self.kernel_size = utils._pair(kernel_size)
        self.order = 4

        # Channels factorization
        self.in_shape = factorize(self.in_channels // self.groups, self.order)
        self.out_shape = factorize(self.out_channels // self.groups, self.order)

        if cr is not None:
            self.rank = self.cr_to_rank(cr)
        else: # if cr is None:
            self.rank = rank

        # print(self.rank)

        # Padding function
        padding = utils._pair(padding)
        padding = utils._pair(padding[0]) + utils._pair(padding[1])

        if padding_mode == "circular":
            _pad = lambda inputs: _circular_pad(inputs, utils._pair(padding))
        elif padding_mode == "zeros":
            _pad = lambda inputs: F.pad(inputs, utils._pair(padding), mode="constant")
        else:  # if padding_mode in ["reflect", "replicate"]:
            _pad = lambda inputs: F.pad(inputs, utils._pair(padding), mode=self.padding_mode)

        # Linear function
        self.conv2d = lambda inputs, weights, bias: \
            F.conv2d(_pad(inputs), weights, bias, stride, utils._pair(0), dilation, groups)

        # CP cores and (optional) bias
        self.kernels = nn.ParameterList()

        for l in range(self.order - 1):
            self.kernels.append(nn.Parameter(torch.Tensor(
                self.in_shape[l], self.out_shape[l], self.rank)))

        self.kernels.append(nn.Parameter(torch.Tensor(
            self.groups, self.kernel_size[0], self.kernel_size[1], self.rank)))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:  # if not bias:
            self.register_parameter('bias', None)

        self.init_params()

        # Einsum equation
        self.equation_l = "iar,jbr->ijabr"
        self.equation_r = "kcr,ghwr->ghwkcr"
        self.equation   = "ijabr,ghwkcr->gijkabchw"

    def cr_to_rank(self, cr):
        """
        Determine the rank based on the compression rate.

        Argument:
        ---------
        cr: float
            The compression rate of the CP decomposition.

        Return:
        -------
        rank: int
            The rank of the CP decomposition.

        """
        rank = np.int(cr * (self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] // self.groups)
            / (self.kernel_size[0] * self.kernel_size[1] * self.groups + np.sum(np.multiply(self.in_shape, self.out_shape))))

        return rank

    def init_params(self):
        """
        Initialization of the CP cores.

        """
        for l in range(self.order - 1):
            fan_in = self.in_shape[l]
            fan_out = self.out_shape[l]
            bound = math.sqrt(6. / (fan_in + fan_out))
            torch.nn.init.uniform_(self.kernels[l], -bound, bound)

        fan_in = self.kernel_size[0] * self.kernel_size[1] * self.rank
        fan_out = self.kernel_size[0] * self.kernel_size[1] * 1
        bound = math.sqrt(6. / (fan_in + fan_out))
        torch.nn.init.uniform_(self.kernels[self.order - 1], -bound, bound)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, inputs):
        """
        Computation of the reshaped CP 2D-convolutional layer.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size
            [batch_size, in_channels, in_height, in_width]
            Input to the reshaped CP 2D-convolutional layer.

        Return:
        -------
        outputs: another 4-th order tensor of size
            [batch_size, in_channels, out_height, out_width]
            Output of the reshaped CP 2D-convolutional layer.

        """
        def calc_forward(inputs):

            weights_l = torch.einsum(self.equation_l, [self.kernels[0], self.kernels[1]])
            weights_r = torch.einsum(self.equation_r, [self.kernels[2], self.kernels[3]])
            weights   = torch.einsum(self.equation, [weights_l, weights_r])

            weights = torch.reshape(weights, (self.out_channels,
                self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]))

            outputs = self.conv2d(inputs, weights, self.bias)
            return outputs


        return checkpoint(calc_forward, inputs)


## Reshaped Tensor-Train 2D-Convolutional Layer
class RTT_O4_Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ranks=None, cr=None,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        """
        Initialization of reshaped Tensor-Train 2D-convolutional layer.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        in_channels:  int
            Number of channels of the input tensor.
        out_channels: int
            Number of channels of the output tensor.

        (Hyper-parameters of the Tensor-Train decomposition)
        kernel_size: int
            The length of the convolutional kernels.
        ranks: int
            The ranks of Tensor-Train decomposition (where all ranks are assumed to be the same).
            default: 8

        (Hyper-parameters of the convolutional operation)
        stride: int
            The stride of convolutional kernels.
            Default: 1
        dilation: int
            The spacing between the elements in the convolutional kernels.
            Default: 1
        padding: int
            The number padding entries added to both side of the input.
            Default: 0
        padding_mode: str ("zeros", "reflect", "replicate" or "circular")
            The padding mode of the convolutional layer.
            Default: "circular"

        (additional hyper-parameters)
        groups: int
            The number of interconnected blocks in the convolutional layer.
            Note: Both in_channels and out_channels should be divisible by groups.
            Default: 1
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True

        """
        super(RTT_O4_Conv2d, self).__init__()

        # Input/output interface
        self.groups = groups

        self.in_channels = in_channels
        self.out_channels = out_channels

        assert self.in_channels % self.groups == 0, \
            "The input channels should be divisble by the number of groups."
        assert self.out_channels % self.groups == 0, \
            "The output channels should be divisble by number of groups."

        # Tensor-Train decomposition
        self.kernel_size = utils._pair(kernel_size)
        self.order = 4

        # Channels factorization
        self.in_shape = factorize(self.in_channels // self.groups, self.order)
        self.out_shape = factorize(self.out_channels // self.groups, self.order)

        if cr is not None:
            self.ranks = self.cr_to_ranks(cr)
        else: # if cr is None:
            self.ranks = ranks

        # Padding function
        padding = utils._pair(padding)
        padding = utils._pair(padding[0]) + utils._pair(padding[1])

        if padding_mode == "circular":
            _pad = lambda inputs: _circular_pad(inputs, utils._pair(padding))
        elif padding_mode == "zeros":
            _pad = lambda inputs: F.pad(inputs, utils._pair(padding), mode="constant")
        else:  # if padding_mode in ["reflect", "replicate"]:
            _pad = lambda inputs: F.pad(inputs, utils._pair(padding), mode=self.padding_mode)

        # Linear function
        self.conv2d = lambda inputs, weights, bias: \
            F.conv2d(_pad(inputs), weights, bias, stride, utils._pair(0), dilation, groups)

        # Tensor-Train cores and (optional) bias
        self.kernels = nn.ParameterList()

        self.kernels.append(nn.Parameter(torch.Tensor(
            self.in_shape[0], self.out_shape[0], self.ranks)))

        for l in range(1, self.order - 1):
            self.kernels.append(nn.Parameter(torch.Tensor(
                self.in_shape[l], self.out_shape[l], self.ranks, self.ranks)))

        self.kernels.append(nn.Parameter(torch.Tensor(
            self.kernel_size[0], self.kernel_size[1], self.groups, self.ranks)))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:  # if not bias:
            self.register_parameter('bias', None)

        self.init_params()

        # Einsum equation
        self.equation_l = "iar,jbrs->ijabs"
        self.equation_r = "kcst,ghwt->ghwkcs"
        self.equation   = "ijabs,ghwkcs->gijkabchw"

    def cr_to_ranks(self, cr):
        """
        Determine the rank based on the compression rate.

        Argument:
        ---------
        cr: float
            The compression rate of the Tensor-Train decomposition.

        Return:
        -------
        rank: int
            The ranks of the Tensor-Train decomposition.

        """
        ranks = np.int(solve_quadratic_equation(
            a = np.sum(np.multiply(self.in_shape[1:], self.out_shape[1:])),
            b = self.in_shape[0] * self.out_shape[0] + self.kernel_size[0] * self.kernel_size[1] * self.groups,
            c = - cr * (self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] // self.groups)
        ))

        return ranks

    def init_params(self):
        """
        Initialization of the Tensor-Train cores.

        """
        fan_in = self.in_shape[0]
        fan_out = self.out_shape[0] * self.ranks
        bound = math.sqrt(6. / (fan_in + fan_out))
        torch.nn.init.uniform_(self.kernels[0], -bound, bound)

        for l in range(1, self.order - 1):
            fan_in = self.in_shape[l] * self.ranks
            fan_out = self.out_shape[l] * self.ranks
            bound = math.sqrt(6. / (fan_in + fan_out))
            torch.nn.init.uniform_(self.kernels[l], -bound, bound)

        fan_in = self.kernel_size[0] * self.kernel_size[1] * self.ranks
        fan_out = self.kernel_size[0] * self.kernel_size[1] * 1
        bound = math.sqrt(6. / (fan_in + fan_out))
        torch.nn.init.uniform_(self.kernels[self.order - 1], -bound, bound)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, inputs):
        """
        Computation of the reshaped Tensor-Train 2D-convolutional layer.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size
            [batch_size, in_channels, in_height, in_width]
            Input to the reshaped Tensor-Train 2D-convolutional layer.

        Return:
        -------
        outputs: another 4-th order tensor of size
            [batch_size, in_channels, out_height, out_width]
            Output of the reshaped Tensor-Train 2D-convolutional layer.

        """
        weights_l = torch.einsum(self.equation_l, [self.kernels[0], self.kernels[1]])
        weights_r = torch.einsum(self.equation_r, [self.kernels[2], self.kernels[3]])
        weights   = torch.einsum(self.equation, [weights_l, weights_r])

        weights = torch.reshape(weights, (self.out_channels,
            self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]))

        outputs = self.conv2d(inputs, weights, self.bias)

        return outputs


## Reshaped Tensor-Ring 2D-Convolutional Layer
class RTR_O4_Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ranks=None, cr=None,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        """
        Initialization of reshaped Tensor-Ring 2D-convolutional layer.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        in_channels:  int
            Number of channels of the input tensor.
        out_channels: int
            Number of channels of the output tensor.

        (Hyper-parameters of the Tensor-Ring decomposition)
        kernel_size: int
            The length of the convolutional kernels.
        order: int
            The order of Tensor-Ring decomposition (i.e. the number of core tensors).
            default: 3
        ranks: int
            The ranks of Tensor-Ring decomposition (where all ranks are assumed to be the same).
            default: 8

        (Hyper-parameters of the convolutional operation)
        stride: int
            The stride of convolutional kernels.
            Default: 1
        dilation: int
            The spacing between the elements in the convolutional kernels.
            Default: 1
        padding: int
            The number padding entries added to both side of the input.
            Default: 0
        padding_mode: str ("zeros", "reflect", "replicate" or "circular")
            The padding mode of the convolutional layer.
            Default: "circular"

        (additional hyper-parameters)
        groups: int
            The number of interconnected blocks in the convolutional layer.
            Note: Both in_channels and out_channels should be divisible by groups.
            Default: 1
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True

        """
        super(RTR_O4_Conv2d, self).__init__()

        # Input/output interface
        self.groups = groups

        self.in_channels = in_channels
        self.out_channels = out_channels

        assert self.in_channels % self.groups == 0, \
            "The input channels should be divisble by the number of groups."
        assert self.out_channels % self.groups == 0, \
            "The output channels should be divisble by number of groups."

        # Tensor-Train decomposition
        self.kernel_size = utils._pair(kernel_size)
        self.order = 4

        # Channels factorization
        self.in_shape = factorize(self.in_channels // self.groups, self.order)
        self.out_shape = factorize(self.out_channels // self.groups, self.order)

        if cr is not None:
            self.ranks = self.cr_to_ranks(cr)
        else: # if cr is None:
            self.ranks = ranks

        # Padding function
        padding = utils._pair(padding)
        padding = utils._pair(padding[0]) + utils._pair(padding[1])

        if padding_mode == "circular":
            _pad = lambda inputs: _circular_pad(inputs, utils._pair(padding))
        elif padding_mode == "zeros":
            _pad = lambda inputs: F.pad(inputs, utils._pair(padding), mode="constant")
        else:  # if padding_mode in ["reflect", "replicate"]:
            _pad = lambda inputs: F.pad(inputs, utils._pair(padding), mode=self.padding_mode)

        # Linear function
        self.conv2d = lambda inputs, weights, bias: \
            F.conv2d(_pad(inputs), weights, bias, stride, utils._pair(0), dilation, groups)

        # Tensor-Train cores and (optional) bias
        self.kernels = nn.ParameterList()

        for l in range(self.order - 1):
            self.kernels.append(nn.Parameter(torch.Tensor(
                self.in_shape[l], self.out_shape[l], self.ranks, self.ranks)))

        self.kernels.append(nn.Parameter(torch.Tensor(
            self.kernel_size[0], self.kernel_size[1], self.groups, self.ranks, self.ranks)))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:  # if not bias:
            self.register_parameter('bias', None)

        self.init_params()

        # Einsum equation
        self.equation_l = "iaur,jbrs->ijabus"
        self.equation_r = "kcst,ghwtu->ghwkcsu"
        self.equation   = "ijabus,ghwkcsu->gijkabchw"

    def cr_to_ranks(self, cr):
        """
        Determine the rank based on the compression rate.

        Argument:
        ---------
        cr: float
            The compression rate of the Tensor-Train decomposition.

        Return:
        -------
        rank: int
            The ranks of the Tensor-Train decomposition.

        """
        ranks = np.int(np.sqrt(cr * (self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] // self.groups)
            / (self.kernel_size[0] * self.kernel_size[1] * self.groups + np.sum(np.multiply(self.in_shape, self.out_shape)))))

        return ranks

    def init_params(self):
        """
        Initialization of the Tensor-Train cores.

        """
        for l in range(self.order - 1):
            fan_in = self.in_shape[l] * self.ranks
            fan_out = self.out_shape[l] * self.ranks
            bound = math.sqrt(6. / (fan_in + fan_out))
            torch.nn.init.uniform_(self.kernels[l], -bound, bound)

        fan_in = self.kernel_size[0] * self.kernel_size[1] * self.ranks
        fan_out = self.kernel_size[0] * self.kernel_size[1] * 1
        bound = math.sqrt(6. / (fan_in + fan_out))
        torch.nn.init.uniform_(self.kernels[self.order - 1], -bound, bound)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, inputs):
        """
        Computation of the reshaped Tensor-Train 2D-convolutional layer.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size
            [batch_size, in_channels, in_height, in_width]
            Input to the reshaped Tensor-Train 2D-convolutional layer.

        Return:
        -------
        outputs: another 4-th order tensor of size
            [batch_size, in_channels, out_height, out_width]
            Output of the reshaped Tensor-Train 2D-convolutional layer.

        """
        weights_l = torch.einsum(self.equation_l, [self.kernels[0], self.kernels[1]])
        weights_r = torch.einsum(self.equation_r, [self.kernels[2], self.kernels[3]])
        weights   = torch.einsum(self.equation, [weights_l, weights_r])

        weights = torch.reshape(weights, (self.out_channels,
            self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]))

        outputs = self.conv2d(inputs, weights, self.bias)

        return outputs


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    batch_size, in_height, in_width = 2, 32, 32

    # print("Testing RCP-Conv2d layer...")
    # for [in_channels, out_channels] in [[32, 256], [64, 512], [128, 1024]]:
    #     for cr in [0.1, 0.2, 0.5, 1.0]:
    #         print("in_channels = %d, out_channels = %d, compression_rate = %.3f"
    #             % (in_channels, out_channels, cr))

    #         base = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1).to(device)

    #         inputs  = torch.randn(batch_size,
    #             in_channels, in_height, in_width).to(device)
    #         module  = RCP_O4_Conv2d(in_channels, out_channels,
    #             kernel_size = 3, cr = cr, padding = 1).to(device)
    #         outputs = module(inputs)

    #         print(sum(param.numel() for param in module.parameters() if param.requires_grad) /
    #             sum(param.numel() for param in base.parameters() if param.requires_grad))

    #         print(torch.min(outputs), torch.max(outputs))


    # print("Testing RTT-Conv2d layer...")
    # for [in_channels, out_channels] in [[32, 256], [64, 512], [128, 1024]]:
    #     for cr in [0.1, 0.2, 0.5, 1.0]:
    #         print("in_channels = %d, out_channels = %d, compression_rate = %.3f"
    #             % (in_channels, out_channels, cr))

    #         base = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1).to(device)

    #         inputs  = torch.randn(batch_size,
    #             in_channels, in_height, in_width).to(device)
    #         module  = RTT_O4_Conv2d(in_channels, out_channels,
    #             kernel_size = 3, cr = cr, padding = 1).to(device)
    #         outputs = module(inputs)

    #         print(sum(param.numel() for param in module.parameters() if param.requires_grad) /
    #             sum(param.numel() for param in base.parameters() if param.requires_grad))

    #         print(torch.min(outputs), torch.max(outputs))


    print("Testing RTR-Conv2d layer...")
    for [in_channels, out_channels] in [[32, 256], [64, 512], [128, 1024]]:
        for cr in [0.1, 0.2, 0.5, 1.0]:
            print("in_channels = %d, out_channels = %d, compression_rate = %.3f"
                % (in_channels, out_channels, cr))

            base = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1).to(device)

            inputs  = torch.randn(batch_size,
                in_channels, in_height, in_width).to(device)
            module  = RTR_O4_Conv2d(in_channels, out_channels,
                kernel_size = 3, cr = cr, padding = 1).to(device)
            outputs = module(inputs)

            print(sum(param.numel() for param in module.parameters() if param.requires_grad) /
                sum(param.numel() for param in base.parameters() if param.requires_grad))

            print(torch.min(outputs), torch.max(outputs))
