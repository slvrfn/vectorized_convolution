import numpy as np
from convolution import Conv2D

in_channels = 3
out_channels = 128
kernel_size = 3
stride = 2
padding = 1
batch_size = (1, in_channels, 12, 10)
dout_size = (1, out_channels, 6, 5)

np.random.seed(42)

x = np.random.random(batch_size)  # create data for forward pass
dout = np.random.random(dout_size)  # create random data for backward
print('x: ', x.shape)
print('d_out: ', dout.shape)

conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding)

conv_out = conv.forward(x)
print('conv_out: ', conv_out.shape)

db, dw, dx = conv.backward(dout)
print('db: ', db.shape)
print('dw: ', dw.shape)
print('dx: ', dx.shape)
