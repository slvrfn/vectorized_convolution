import numpy as np


def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )


class Conv2D:
    """
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters.
        """

        n, c, h, w = x.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        windows = getWindows(x, (n, c, out_h, out_w), self.kernel_size, self.padding, self.stride)

        out = np.einsum('bihwkl,oikl->bohw', windows, self.weight)

        # add bias to kernels
        out += self.bias[None, :, None, None]

        self.cache = x, windows
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: dx, dw, and db relative to this module
        """
        x, windows = self.cache

        padding = self.kernel_size - 1 if self.padding == 0 else self.padding

        dout_windows = getWindows(dout, x.shape, self.kernel_size, padding=padding, stride=1, dilate=self.stride - 1)
        rot_kern = np.rot90(self.weight, 2, axes=(2, 3))

        db = np.sum(dout, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', windows, dout)
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        return db, dw, dx
