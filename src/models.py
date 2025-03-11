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
        
        self.initialize_weights()

    def initialize_weights(self):
        conv_layers = [self.conv1, self.conv2, self.conv3]
        activations = [self.act1, self.act2, self.act3]

        for conv, act in zip(conv_layers, activations):
            if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
                nn.init.xavier_uniform_(conv.weight)
            else:
                nn.init.kaiming_normal_(conv.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x
    
class overlapping_CNN(nn.Module):
    def __init__(self, act1, act2, act3):
        super(overlapping_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=3, padding=0, bias=False)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=2, stride=2, padding=0, bias=False)  

        self.act1 = act1
        self.act2 = act2
        self.act3 = act3
        
        self.initialize_weights()

    def initialize_weights(self):
        conv_layers = [self.conv1, self.conv2, self.conv3]
        activations = [self.act1, self.act2, self.act3]

        for conv, act in zip(conv_layers, activations):
            if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
                nn.init.xavier_uniform_(conv.weight)
            else:
                nn.init.kaiming_normal_(conv.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x

class FCNN(nn.Module):
    def __init__(self, act1, act2, act3):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(12, 4, bias=False)
        self.fc2 = nn.Linear(4, 2, bias=False)
        self.fc3 = nn.Linear(2, 1, bias=False)

        self.act1 = act1
        self.act2 = act2
        self.act3 = act3
        
        self.initialize_weights()

    def initialize_weights(self):
        fc_layers = [self.fc1, self.fc2, self.fc3]
        activations = [self.act1, self.act2, self.act3]

        for fc, act in zip(fc_layers, activations):
            if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
                nn.init.kaiming_normal_(fc.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
                nn.init.xavier_uniform_(fc.weight)
            else:
                nn.init.kaiming_normal_(fc.weight)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return x

# Finding: using a different stride than the teacher doesn't work!
class overlapping_CNN2(nn.Module):
    def __init__(self, act1, act2, act3):
        super(overlapping_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)  

        self.act1 = act1
        self.act2 = act2
        self.act3 = act3
        
        self.initialize_weights()

    def initialize_weights(self):
        conv_layers = [self.conv1, self.conv2, self.conv3]
        activations = [self.act1, self.act2, self.act3]

        for conv, act in zip(conv_layers, activations):
            if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
                nn.init.xavier_uniform_(conv.weight)
            else:
                nn.init.kaiming_normal_(conv.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x