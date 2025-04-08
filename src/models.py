import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

# TODO: update when using GeLU
class BaseCNN(nn.Module):
    """Base class for CNN models with configurable activation functions."""
    
    def __init__(self, layers_config, activations, config_path=None):
        super(BaseCNN, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.Conv1d(in_c, out_c, kernel, stride, padding=0, bias=False)
            for in_c, out_c, kernel, stride in layers_config
        ])
        
        self.activations = activations
        self.config = utils.read_config(config_path)

        init = self.config.get("init")
        if init:
            print("Initializing the weights with scaling factor:", init)
            self.initialize_weights(init)

    def initialize_weights(self, init):
        """Applies weight initialization based on activation functions."""
        for layer, act in zip(self.layers, self.activations):
            if act == torch.relu:
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                layer.weight.data = layer.weight.data * init
            elif act == torch.sigmoid:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
                layer.weight.data = layer.weight.data * init
            elif act == torch.tanh:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                layer.weight.data = layer.weight.data * init
            else:
                nn.init.kaiming_normal_(layer.weight)
    
    # def initialize_weights(self, init):
    #     """Applies weight initialization based on activation functions."""
    #     for layer, act in zip(self.layers, self.activations):
    #         if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
    #             nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
    #             nn.init.xavier_uniform_(layer.weight)
    #         else:
    #             nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x

class NonOverlappingCNN(BaseCNN):
    """CNN with non-overlapping strides."""
    def __init__(self, act1, act2, act3, config_path=None):
        layers_config = [
            (1, 1, 3, 3), #in_c, out_c, kernel_size, stride
            (1, 1, 2, 2),
            (1, 1, 2, 2)
        ]
        super().__init__(layers_config, [act1, act2, act3], config_path)

class OverlappingCNN(BaseCNN):
    """CNN with overlapping strides."""
    def __init__(self, act1, act2, act3, config_path=None):
        layers_config = [
            (1, 4, 3, 3), #in_c, out_c, kernel_size, stride
            (4, 4, 2, 2),
            (4, 1, 2, 2)
        ]
        super().__init__(layers_config, [act1, act2, act3], config_path)

class OverlappingCNN2(BaseCNN):
    """Alternative overlapping CNN with different stride settings."""
    def __init__(self, act1, act2, act3, config_path=None):
        layers_config = [
            (1, 1, 3, 1), #in_c, out_c, kernel_size, stride
            (1, 1, 2, 1),
            (1, 1, 2, 1)
        ]
        super().__init__(layers_config, [act1, act2, act3], config_path)

##########################################################################

class MultiWeightCNN(nn.Module):
    """CNN with alternating weights for each chunk of input and configurable activations."""
    def __init__(self, act1, act2, act3, config_path=None):
        self.activations = [act1, act2, act3]
        super().__init__()
        self.layer1_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([[2.59, -2.83, 0.87]])),
            nn.Parameter(torch.tensor([[-1.22, 0.45, 0.88]]))
            #nn.Parameter(torch.tensor([[1.45, -0.92, 0.66]]))
        ])
        self.layer2_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([[-1.38, 1.29]])),
            nn.Parameter(torch.tensor([[0.35, -0.73]]))
        ])
        self.layer3_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([[0.86, -0.84]])),
        ])

    def forward(self, x):
        # Layer 1: input (batch, 12) ➝ 4 chunks of 3 ➝ (batch, 4)
        chunks1 = x.unfold(1, size=3, step=3)  # (batch, 4, 3)
        out1 = []
        for i in range(chunks1.size(1)):
            weight = self.layer1_weights[i % 2]
            out1.append(self.activations[0](F.linear(chunks1[:, i], weight)))  # (batch, 1)
        x1 = torch.cat(out1, dim=1)

        # Layer 2: input (batch, 4) ➝ 2 chunks of 2 ➝ (batch, 2)
        chunks2 = x1.unfold(1, size=2, step=2)
        out2 = []
        for i in range(chunks2.size(1)):
            weight = self.layer2_weights[i % 2]
            out2.append(self.activations[1](F.linear(chunks2[:, i], weight)))
        x2 = torch.cat(out2, dim=1)

        # Layer 3: input (batch, 2) ➝ 1 chunk of 2 ➝ (batch, 1)
        weight = self.layer3_weights[0]  # or still alternate if more chunks
        x3 = self.activations[2](F.linear(x2, weight))

        return x3.unsqueeze(1)

##########################################################################

class BaseFCNN(nn.Module):
    """Base class for Fully Connected Neural Networks with configurable architecture."""
    
    def __init__(self, layer_sizes, activations, config_path):
        super(BaseFCNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_f, out_f, bias=False) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        self.activations = activations
        self.config = utils.read_config(config_path)

        init = self.config.get("init")
        if init:
            print("Initializing the weights with scaling factor:", init)
            self.initialize_weights(init)

    def initialize_weights(self, init):
        """Initialize weights based on activation functions."""
        for layer, act in zip(self.layers, self.activations):
            if act == torch.relu:
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                layer.weight.data = layer.weight.data * init
            elif act == torch.sigmoid:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
                layer.weight.data = layer.weight.data * init
            elif act == torch.tanh:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                layer.weight.data = layer.weight.data * init
            else:
                nn.init.kaiming_normal_(layer.weight)

    # def initialize_weights(self, init):
    #     """Applies weight initialization based on activation functions."""
    #     for layer, act in zip(self.layers, self.activations):
    #         if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
    #             nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
    #             nn.init.xavier_uniform_(layer.weight)
    #         else:
    #             nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x


class FCNN(BaseFCNN):
    """Fully Connected Neural Network with uniform layer sizes."""
    
    def __init__(self, act1, act2, act3, config_path):
        super().__init__([12, 128, 128, 1], [act1, act2, act3], config_path)


class FCNN_decreasing(BaseFCNN):
    """Fully Connected Neural Network with decreasing layer sizes."""
    
    def __init__(self, act1, act2, act3, config_path):
        super().__init__([12, 256, 32, 1], [act1, act2, act3], config_path)