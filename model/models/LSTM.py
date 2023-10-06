import torch.nn as nn

class LSTM(nn.Module):
    """
        Parameters:
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size=49, hidden_size=64, output_size=8, num_layers=3):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.linear1 = nn.Linear(hidden_size, output_size)
 
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear1(lstm_out)
        return linear_out
        
