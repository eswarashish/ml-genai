import torch
import torch.nn as nn
from torch import device
from torch import dtype
from torch import Tensor

class ETDLSTM(nn.Module):
    def __init__(self,input_size:int,hidden_size:int,output_size:int,num_layers: int,dropout:float,device:device,dtype: dtype):
        super(ETDLSTM,self).__init__()
        self.input = input_size
        self.output = output_size
        self.hidden  = hidden_size
        self.num = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,batch_first=True,device=device,dtype=dtype)
        self.sequential = nn.Sequential(nn.Linear(hidden_size,output_size,device=device,dtype=dtype),
                                        nn.SELU(),
                                        nn.Linear(output_size,output_size,device=device,dtype=dtype))


    def forward(self,x:Tensor):
        h0 = torch.zeros(size=[self.num,x.size(0),self.hidden],device=self.device)
        
        c0 = torch.zeros(size=[self.num,x.size(0),self.hidden],device=self.device)
        
        out, (_,_) = self.lstm(x,(h0 ,c0))

        return self.sequential(out)
    
    
