import torch
from torch import nn, optim
import torch.nn.functional as F


class IE_block(nn.Module):
    def __init__(self,input_num,out_num,IE_Input_len):
        super(IE_block, self).__init__()
        self.IE_Input_len = IE_Input_len
        self.output = nn.Linear(input_num, out_num)
    def forward(self, x):
        x = x.reshape((x.shape[0],x.shape[1],x.shape[2],1))
        ###piecewise sampling
        x = IE_block.piecewise_sample(x,self.IE_Input_len)
        ### Dimension transformation
        x = self.output(x)
        return x

    @staticmethod
    def piecewise_sample(data,n):
        result = 0.0
        data_len = data.shape[2] // n
        for i in range(n):
            line = data[:,:,data_len*i:data_len*(i+1),:]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result
