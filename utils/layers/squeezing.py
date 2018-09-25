import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Squeezing(nn.Module):
    def __init__(self,filterSize = 2):
        super(Squeezing,self).__init__()
        self.filterSize = filterSize
    def forward(self,input):
        scale_factor = self.filterSize
        batch_size, in_channels, in_height, in_width = input.size()

        out_channels = int(in_channels // (scale_factor * scale_factor))
        out_height = int(in_height * scale_factor)
        out_width = int(in_width * scale_factor)

        if scale_factor >= 1:
            input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor,in_height, in_width)
            shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        else:
            block_size = int(1 / scale_factor)
            input_view = input.contiguous().view(batch_size, in_channels, out_height, block_size,out_width, block_size)
            shuffle_out = input_reshape.permute(0, 1, 3, 5, 2, 4).contiguous()

        return shuffle_out.reshape(batch_size, out_channels, out_height, out_width)
