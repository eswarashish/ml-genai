from torch import nn
from torch import device
from torch import dtype
from torch import Tensor
import torch
class VanillaLSTM(nn.Module):
    def __init__(self,input_size: int, num_layers:int , device: device, output_size: int, droupout: float,dtype: dtype):
           super(VanillaLSTM,self).__init__()
           self.input_size = input_size
           self.num_layers = num_layers
           self.output_size = output_size
           self.device = device
           self.lstm = nn.LSTM(input_size=self.input_size,hidden_size=self.output_size,device=self.device,batch_first=True,dropout=droupout,dtype=dtype)

    def forward(self,x:Tensor):
          h0 = torch.zeros([self.num_layers,x.size(0),self.output_size]) 
          
          c0 = torch.zeros([self.num_layers,x.size(0),self.output_size])

          out = self.lstm(x,(h0,c0))

          return out