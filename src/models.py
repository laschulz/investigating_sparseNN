import torch.nn as nn
import utils
    
class BaseCNN(nn.Module):
    """Base class for CNN models with configurable activation functions."""
    
    def __init__(self, layers_config, activations):
        super(BaseCNN, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.Conv1d(in_c, out_c, kernel, stride, padding=0, bias=False)
            for in_c, out_c, kernel, stride in layers_config
        ])
        
        self.activations = activations
        self.config = utils.read_config()

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
        x = x.unsqueeze(1)
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x

# Subclasses inheriting from BaseCNN

class NonOverlappingCNN(BaseCNN):
    """CNN with non-overlapping strides."""
    def __init__(self, act1, act2, act3):
        layers_config = [
            (1, 1, 3, 3),
            (1, 1, 2, 2),
            (1, 1, 2, 2)
        ]
        super().__init__(layers_config, [act1, act2, act3])

class OverlappingCNN(BaseCNN):
    """CNN with overlapping strides."""
    def __init__(self, act1, act2, act3):
        layers_config = [
            (1, 4, 3, 3),
            (4, 4, 2, 2),
            (4, 1, 2, 2)
        ]
        super().__init__(layers_config, [act1, act2, act3])

class OverlappingCNN2(BaseCNN):
    """Alternative overlapping CNN with different stride settings."""
    def __init__(self, act1, act2, act3):
        layers_config = [
            (1, 1, 3, 1),
            (1, 1, 2, 1),
            (1, 1, 2, 1)
        ]
        super().__init__(layers_config, [act1, act2, act3])

##########################################################################

class FCNN(nn.Module):
    """Fully Connected Neural Network."""
    
    def __init__(self, act1, act2, act3):
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(12, 4, bias=False),
            nn.Linear(4, 2, bias=False),
            nn.Linear(2, 1, bias=False)
        ])
        
        self.activations = [act1, act2, act3]
        self.config = utils.read_config()

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
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x

class Transformer(nn.Module):
    """Placeholder for transformer model."""
    pass