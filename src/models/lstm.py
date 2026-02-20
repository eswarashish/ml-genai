import torch.nn.functional as F
import torch.nn as nn

class ETDLSTM(nn.Module):
    def __init__(self,input_size):
        super(ETDLSTM,self).__init__()

        self.lstm = nn.LSTM()