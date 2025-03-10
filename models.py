import torch.nn as nn

class nonoverlapping_CNN(nn.Module):
    def __init__(self, act1, act2, act3):
        super(nonoverlapping_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3, padding=0, bias = False)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, bias = False)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, bias = False)

        self.act1 = act1
        self.act2 = act2
        self.act3 = act3
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x