import torch
from torch import nn, optim
import torch.nn.functional as F


class DF_block(nn.Module):
    def __init__(self,num_ga,out_len):
        super(DF_block, self).__init__()
        self.att1 = RF_att(num_ga)
        self.att2 = RF_att(num_ga)
        self.att3 = RF_att(num_ga)
        self.out_len = out_len//4

    def forward(self, x):

        line1 = x[:,:,0:self.out_len,:]
        line1 = self.att1(line1)
        line1 = line1.sum(dim=-1)

        line2 = x[:,:,self.out_len:self.out_len*2,:]
        line2 = self.att1(line2)
        line2 = line2.sum(dim=-1)

        line3 = x[:,:,self.out_len*2:,:]
        line3 = self.att1(line3)
        line3 = line3.sum(dim=-1)
        x = torch.cat([line1, line2,line3], dim=2)

        return x

class RF_att(nn.Module):
    def __init__(self, dim_input):
        super(RF_att, self).__init__()
        self.QK = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q_K = self.QK(x)
        Q_K = self.softmax(Q_K)
        x = x * Q_K
        return x