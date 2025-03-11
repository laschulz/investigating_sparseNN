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
        
        self._initialize_weights()

    def _initialize_weights(self):
        conv_layers = [self.conv1, self.conv2, self.conv3]
        activations = [self.act1, self.act2, self.act3]

        for conv, act in zip(conv_layers, activations):
            if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
                nn.init.xavier_uniform_(conv.weight)
            else:
                nn.init.kaiming_normal_(conv.weight)  # Default to Kaiming for unknown activations

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x