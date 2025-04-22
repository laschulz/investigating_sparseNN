import json
import torch

import models

def create_model(model_type, config_path=None):
    model_dict = {
        "baselineCNN_tanh": lambda: models.BaselineCNN(torch.tanh, torch.tanh, torch.tanh, config_path),
        "baselineCNN_relu": lambda: models.BaselineCNN(torch.relu, torch.relu, torch.relu, config_path),
        "baselineCNN_gelu": lambda: models.BaselineCNN(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, config_path),
        "baselineCNN_sigmoid": lambda: models.BaselineCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid, config_path),

        "splitFilterCNN_tanh": lambda: models.SplitFilterCNN(torch.tanh, torch.tanh, torch.tanh, config_path),
        "splitFilterCNN_relu": lambda: models.SplitFilterCNN(torch.relu, torch.relu, torch.relu, config_path),
        "splitFilterCNN_gelu": lambda: models.SplitFilterCNN(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, config_path),
        "splitFilterCNN_sigmoid": lambda: models.SplitFilterCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid, config_path),
        
        "multiChannelCNN_tanh": lambda: models.MultiChannelCNN(torch.tanh, torch.tanh, torch.tanh, config_path),
        "multiChannelCNN_relu": lambda: models.MultiChannelCNN(torch.relu, torch.relu, torch.relu, config_path),
        "multiChannelCNN_gelu": lambda: models.MultiChannelCNN(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, config_path),
        "multiChannelCNN_sigmoid": lambda: models.MultiChannelCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid, config_path),
        
        "fcn_128_128_tanh": lambda: models.FCN_128_128(torch.tanh, torch.tanh, torch.tanh, config_path),
        "fcn_128_128_relu": lambda: models.FCN_128_128(torch.relu, torch.relu, torch.relu, config_path),
        "fcn_128_128_gelu": lambda: models.FCN_128_128(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, config_path),
        "fcn_128_128_sigmoid": lambda: models.FCN_128_128(torch.sigmoid, torch.sigmoid, torch.sigmoid, config_path),
        
        "fcn_256_32_tanh": lambda: models.FCN_256_32(torch.tanh, torch.tanh, torch.tanh, config_path),
        "fcn_256_32_relu": lambda: models.FCN_256_32(torch.relu, torch.relu, torch.relu, config_path),
        "fcn_256_32_gelu": lambda: models.FCN_256_32(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, config_path),
        "fcn_256_32_sigmoid": lambda: models.FCN_256_32(torch.sigmoid, torch.sigmoid, torch.sigmoid, config_path),

        "fcn_1024_128_relu": lambda: models.FCN_1024_128(torch.relu, torch.relu, torch.relu, config_path)
    }
    return model_dict[model_type]()


def read_config(config_path):
    print(config_path)
    full_path = f"config_files/{config_path}"
    with open(full_path, "r") as file:
        config = json.load(file)
    return config

def init_teacher(teacher_model, teacher_name):
    if "baselineCNN" in teacher_name:
        teacher_weights = [
            torch.tensor([[[2.59, -2.83, 0.87]]]),  # conv1
            torch.tensor([[[-1.38, 1.29]]]),        # conv2
            torch.tensor([[[0.86, -0.84]]])         # conv3
        ]
        with torch.no_grad():
            for layer, weight in zip(teacher_model.layers, teacher_weights):
                layer.weight.copy_(weight)
    elif "multiChannelCNN" in teacher_name:
        teacher_weights = [
        torch.tensor([[[-0.78, -0.12,  0.70]],
                    [[-1.16,  0.47,  0.05]],
                    [[-0.73,  1.96, -1.01]],
                    [[-0.32,  0.21,  0.63]]]),  # conv1
        torch.tensor([[[ 0.00,  0.04],
                    [ 0.68,  0.34],
                    [ 0.54, -0.22],
                    [-0.14, -0.33]],
                    [[-0.14,  1.59],
                    [ 1.48, -0.52],
                    [-1.26,  0.30],
                    [-0.40, -1.09]],
                    [[-0.71,  0.44],
                    [-0.02, -0.14],
                    [ 0.37, -0.70],
                    [-0.83, -0.38]],
                    [[ 0.89, -0.48],
                    [-0.27, -0.81],
                    [ 1.76, -0.41],
                    [ 0.15,  0.49]]]),  # conv2
        torch.tensor([[[-0.54,  0.16],
                    [-0.74, -0.46],
                    [ 0.08,  0.18],
                    [-0.22,  0.81]]])           # conv3
        ]
        with torch.no_grad():
            for layer, weight in zip(teacher_model.layers, teacher_weights):
                layer.weight.copy_(weight)
    return teacher_model