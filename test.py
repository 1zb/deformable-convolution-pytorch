import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import ConvOffset2d

num_deformable_group = 1

N, inC, inH, inW = 1, 3, 512, 512
outC, outH, outW = 4, 512, 512
kH, kW = 3, 3

conv = nn.Conv2d(
    inC,
    num_deformable_group * 2 * kH * kW,
    kernel_size=(kH, kW),
    stride=(1, 1),
    padding=(1, 1),
    bias=False).cuda()
conv_offset2d = ConvOffset2d(inC, outC, (kH, kW), stride=1, padding=1).cuda()

inputs = Variable(torch.randn(N, inC, inH, inW).cuda())
offset = conv(inputs)
output = conv_offset2d(inputs, offset)
output.backward(output.data)
print(output.size())
