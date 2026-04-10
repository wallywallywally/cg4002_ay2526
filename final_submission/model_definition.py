import torch
import torch.nn as nn

# Input: (30 features, 25 rows)
class CNN1DClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=7),      # 30 inputs
            nn.LeakyReLU(0.1),

            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)        # 9 gestures

    def forward(self, x):
        x = self.conv_block(x)
        x = x.squeeze(-1)
        return self.fc(x)