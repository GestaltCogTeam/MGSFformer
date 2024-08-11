import torch
from torch import nn, optim
import torch.nn.functional as F


class STA_block_att(nn.Module):
    def __init__(self,Input_len, num_id,IE_dim,out_len, dropout, num_head):
        super(STA_block_att, self).__init__()
        self.Time_att = Time_att(Input_len,IE_dim,dropout,num_head)
        self.space_att = space_att2(num_id, dropout, num_head)
        self.cross_att = cross_att(Input_len,IE_dim,dropout,num_head)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Conv1d(in_channels=Input_len * IE_dim, out_channels=out_len, kernel_size=1)
    def forward(self, x):
        x = self.cross_att(self.Time_att(x),self.space_att(x))
        x = x.reshape((x.shape[0],x.shape[1],-1))
        x = self.linear(x.transpose(-2,-1))
        return x.transpose(-2,-1)

### temporal attention
class Time_att(nn.Module):
    def __init__(self, Input_len,dim_input,dropout,num_head):
        super(Time_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.laynorm = nn.LayerNorm([Input_len])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head,1)
    def forward(self, x):
        x = x.permute(0,3,1,2)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-3, -2)
            k = self.dropout(self.key(x)).permute(0,2,3,1)
            v = self.dropout(self.value(x)).transpose(-3, -2)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32)/self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x + result.transpose(-3, -2)
        x = self.laynorm(x)
        return x

### space_attention
class space_att2(nn.Module):
    def __init__(self, dim_input,dropout,num_head):
        super(space_att2, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        x = x.permute(0,2,3,1)
        result = 0.0
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)

        for i in range(self.num_head):

            line = self.dropout(self.softmax(q@k/kd))@ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.permute(0,2,3,1)
        return result

### cross attention
class cross_att(nn.Module):
    def __init__(self, Input_len,dim_input,dropout,num_head):
        super(cross_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.laynorm = nn.LayerNorm([Input_len])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head,1)
    def forward(self, x, x2):
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).transpose(-3, -2)
            k = self.dropout(self.key(x)).transpose(-3, -2)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -2)

            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32)/self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -2) + result
        x = self.laynorm(x)
        return x