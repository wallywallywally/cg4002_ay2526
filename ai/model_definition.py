import torch
import torch.nn as nn

# 1D CNN -> comparable to RNN with less overhead
# Input: (8 features, 25 rows)
class CNN1DClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1DClassifier, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)        # expects 64 x 8 = 512

    def forward(self, x):
        x = self.conv_block(x)
        x = x.squeeze(-1)
        return self.fc(x)
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
# 3-layer MLP -> worst
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)