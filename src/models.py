import torch.nn as nn
import utils
    
class BaseCNN(nn.Module):
    """Base class for CNN models with configurable activation functions."""
    
    def __init__(self, layers_config, activations, device='cpu', config_path=None):
        super(BaseCNN, self).__init__()
        
        self.device = device
        self.layers = nn.ModuleList([
            nn.Conv1d(in_c, out_c, kernel, stride, padding=0, bias=False).to(device)
            for in_c, out_c, kernel, stride in layers_config
        ]).to(device)
        
        self.activations = activations
        self.config = utils.read_config(config_path)

        if self.config.get("init"):
            self.initialize_weights()

    def initialize_weights(self):
        """Applies weight initialization based on activation functions."""
        for layer, act in zip(self.layers, self.activations):
            if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        x = x.to(self.device)
        x = x.unsqueeze(1)
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x

# Subclasses inheriting from BaseCNN

class NonOverlappingCNN(BaseCNN):
    """CNN with non-overlapping strides."""
    def __init__(self, act1, act2, act3, device='cpu', config_path=None):
        layers_config = [
            (1, 1, 3, 3), #in_c, out_c, kernel_size, stride
            (1, 1, 2, 2),
            (1, 1, 2, 2)
        ]
        super().__init__(layers_config, [act1, act2, act3], device, config_path)

class OverlappingCNN(BaseCNN):
    """CNN with overlapping strides."""
    def __init__(self, act1, act2, act3, device='cpu', config_path=None):
        layers_config = [
            (1, 4, 3, 3), #in_c, out_c, kernel_size, stride
            (4, 4, 2, 2),
            (4, 1, 2, 2)
        ]
        super().__init__(layers_config, [act1, act2, act3], device, config_path)

class OverlappingCNN2(BaseCNN):
    """Alternative overlapping CNN with different stride settings."""
    def __init__(self, act1, act2, act3, device='cpu', config_path=None):
        layers_config = [
            (1, 1, 3, 1), #in_c, out_c, kernel_size, stride
            (1, 1, 2, 1),
            (1, 1, 2, 1)
        ]
        super().__init__(layers_config, [act1, act2, act3], device, config_path)

##########################################################################

class BaseFCNN(nn.Module):
    """Base class for Fully Connected Neural Networks with configurable architecture."""
    
    def __init__(self, layer_sizes, activations, device, config_path):
        super(BaseFCNN, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([
            nn.Linear(in_f, out_f, bias=False) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        self.activations = activations
        self.config = utils.read_config(config_path)

        if self.config.get("init"):
            self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights based on activation functions."""
        for layer, act in zip(self.layers, self.activations):
            if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        x = x.to(self.device)
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x


class FCNN(BaseFCNN):
    """Fully Connected Neural Network with uniform layer sizes."""
    
    def __init__(self, act1, act2, act3, device, config_path):
        super().__init__([12, 128, 128, 1], [act1, act2, act3], device, config_path)


class FCNN_decreasing(BaseFCNN):
    """Fully Connected Neural Network with decreasing layer sizes."""
    
    def __init__(self, act1, act2, act3, device, config_path):
        super().__init__([12, 256, 32, 1], [act1, act2, act3], device, config_path)


# class FCNN(nn.Module):
#     """Fully Connected Neural Network."""
    
#     def __init__(self, act1, act2, act3, device):
#         super(FCNN, self).__init__()
#         self.device = device
#         self.layers = nn.ModuleList([
#             nn.Linear(12, 128, bias=False).to(self.device),
#             nn.Linear(128, 128, bias=False).to(self.device),
#             nn.Linear(128, 1, bias=False).to(self.device)
#         ])
        
#         self.activations = [act1, act2, act3]
#         self.config = utils.read_config()

#         if self.config.get("init"):
#             self.initialize_weights()

#     def initialize_weights(self):
#         """Applies weight initialization based on activation functions."""
#         for layer, act in zip(self.layers, self.activations):
#             if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
#                 nn.init.xavier_uniform_(layer.weight)
#             else:
#                 nn.init.kaiming_normal_(layer.weight)

#     def forward(self, x):
#         x = x.to(self.device)
#         for layer, act in zip(self.layers, self.activations):
#             x = act(layer(x))
#         return x