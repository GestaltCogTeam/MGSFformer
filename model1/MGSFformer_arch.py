import torch
from torch import nn

from .IE_block import IE_block
from .STA_block import STA_block_att
from .DF_block import DF_block
from .revin import RevIN


class MGSFformer(nn.Module):
    def __init__(self, Input_len, out_len, num_id, IE_dim, dropout, num_head):
        """
        Input_len: Historical length
        out_len: Future length
        num_id: Number of time series
        IE_dim: Embedding size
        dropout: Droupout
        num_head: Number of multi-head attention
        """
        super(MGSFformer, self).__init__()

        self.RevIN = RevIN(num_id)

        ###RD-block
        self.IE_Input_len = Input_len // 24
        self.IE_block1 = IE_block(1,IE_dim,self.IE_Input_len)
        self.IE_block2 = IE_block(2,IE_dim,self.IE_Input_len)
        self.IE_block3 = IE_block(4, IE_dim, self.IE_Input_len)
        self.IE_block4 = IE_block(8, IE_dim, self.IE_Input_len)
        self.IE_block5 = IE_block(24,IE_dim,self.IE_Input_len)

        self.lay_norm1 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm2 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm3 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm4 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm5 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])

        ###STA-block
        self.ST_block1 = STA_block_att(self.IE_Input_len, num_id,IE_dim,out_len, dropout, num_head)
        self.ST_block2 = STA_block_att(self.IE_Input_len, num_id,IE_dim,out_len, dropout, num_head)
        self.ST_block3 = STA_block_att(self.IE_Input_len, num_id,IE_dim,out_len, dropout, num_head)
        self.ST_block4 = STA_block_att(self.IE_Input_len, num_id,IE_dim,out_len, dropout, num_head)
        self.ST_block5 = STA_block_att(self.IE_Input_len, num_id,IE_dim,out_len, dropout, num_head)

        ###DF_block
        self.DF_block = DF_block(5,out_len)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,**kwargs) -> torch.Tensor:

        # Input [B,H,N,1]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N,1]: B is batch size. N is the number of variables. L is the future length

        x = history_data[:, :, :, 0]
        x = self.RevIN(x, 'norm').transpose(-2, -1)

        x_day = MGSFformer.Get_Coarse_grain(x,24)
        x_12h = MGSFformer.Get_Coarse_grain(x,12)
        x_6h = MGSFformer.Get_Coarse_grain(x, 6)
        x_3h = MGSFformer.Get_Coarse_grain(x, 3)

        ### RD-block
        x_day = self.IE_block1(x_day)
        x_12h = self.IE_block2(x_12h)
        x_6h = self.IE_block3(x_6h)
        x_3h = self.IE_block4(x_3h)
        x = self.IE_block5(x)

        x_day = self.lay_norm1(x_day)
        x_12h = self.lay_norm2(x_12h)
        x_6h = self.lay_norm1(x_6h)
        x_3h = self.lay_norm2(x_3h)
        x = self.lay_norm3(x)

        x_12h = x_12h - x_day
        x_6h = x_6h - x_12h
        x_3h = x_3h - x_6h
        x = x - x_3h

        ### STA-block
        x_day = self.ST_block1(x_day)
        x_12h = self.ST_block2(x_12h)
        x_6h = self.ST_block3(x_6h)
        x_3h = self.ST_block4(x_3h)
        x = self.ST_block5(x)

        ### DF_block
        x_day = x_day.unsqueeze(-1)
        x_12h = x_12h.unsqueeze(-1)
        x_6h = x_6h.unsqueeze(-1)
        x_3h = x_3h.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = torch.cat([x_day,x_12h,x_6h,x_3h, x],dim=-1)
        x = self.DF_block(x).transpose(-2,-1)
        x = self.RevIN(x, 'denorm').unsqueeze(-1)
        return x

    @staticmethod
    def Get_Coarse_grain(data,n):
        result = 0.0
        for i in range(n):
            line = data[:,:,i::n]
            if i == 0:
                result = line
            else:
                result = result + line
        result = result/n
        return result
