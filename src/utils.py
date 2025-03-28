import json
import torch
import re

import models
import transformer_model

def create_model(model_type, device="cpu"):
    model_dict = {
        "nonoverlapping_CNN_all_tanh": lambda: models.NonOverlappingCNN(torch.tanh, torch.tanh, torch.tanh, device=device),
        "nonoverlapping_CNN_all_relu": lambda: models.NonOverlappingCNN(torch.relu, torch.relu, torch.relu, device=device),
        "nonoverlapping_CNN_all_gelu": lambda: models.NonOverlappingCNN(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, device=device),
        "nonoverlapping_CNN_all_sigmoid": lambda: models.NonOverlappingCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid, device=device),
        
        "overlapping_CNN_all_tanh": lambda: models.OverlappingCNN(torch.tanh, torch.tanh, torch.tanh, device=device),
        "overlapping_CNN_all_relu": lambda: models.OverlappingCNN(torch.relu, torch.relu, torch.relu, device=device),
        "overlapping_CNN_all_gelu": lambda: models.OverlappingCNN(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, device=device),
        "overlapping_CNN_all_sigmoid": lambda: models.OverlappingCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid, device=device),
        
        "fcnn_all_tanh": lambda: models.FCNN(torch.tanh, torch.tanh, torch.tanh, device=device),
        "fcnn_all_relu": lambda: models.FCNN(torch.relu, torch.relu, torch.relu, device=device),
        "fcnn_all_gelu": lambda: models.FCNN(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, device=device),
        "fcnn_all_sigmoid": lambda: models.FCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid, device=device),
        
        "fcnn_decreasing_all_tanh": lambda: models.FCNN_decreasing(torch.tanh, torch.tanh, torch.tanh, device=device),
        "fcnn_decreasing_all_relu": lambda: models.FCNN_decreasing(torch.relu, torch.relu, torch.relu, device=device),
        "fcnn_decreasing_all_gelu": lambda: models.FCNN_decreasing(torch.nn.functional.gelu, torch.nn.functional.gelu, torch.nn.functional.gelu, device=device),
        "fcnn_decreasing_all_sigmoid": lambda: models.FCNN_decreasing(torch.sigmoid, torch.sigmoid, torch.sigmoid, device=device),
        
        "nonoverlapping_transformer_tanh": lambda: transformer_model.NonOverlappingViT(torch.tanh, device=device),
        "nonoverlapping_transformer_relu": lambda: transformer_model.NonOverlappingViT(torch.relu, device=device),
        "nonoverlapping_transformer_gelu": lambda: transformer_model.NonOverlappingViT(torch.nn.functional.gelu, device=device),
        "nonoverlapping_transformer_sigmoid": lambda: transformer_model.NonOverlappingViT(torch.sigmoid, device=device)
    }
    return model_dict[model_type]()


def read_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)   
    return config

def init_teacher(teacher_model, teacher_name):
    teacher_weights = []
    if "nonoverlappingCNN" in teacher_name:
        teacher_weights = [
            torch.tensor([[[2.59, -2.83, 0.87]]]),  # conv1
            torch.tensor([[[-1.38, 1.29]]]),        # conv2
            torch.tensor([[[0.86, -0.84]]])         # conv3
        ]
    elif "overlappingCNN" in teacher_name:
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