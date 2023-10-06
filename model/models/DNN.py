import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(49, 64),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(64, 8), 
        )
    
    def forward(self, X):
        X = self.model(X)
        return X





