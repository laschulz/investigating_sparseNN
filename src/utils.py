import json
import torch

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


def read_config():
    with open("config.json", "r") as file:
        config = json.load(file)   
    return config