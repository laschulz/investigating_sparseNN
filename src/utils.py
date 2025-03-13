import json, os
import torch

import models

model = {
    "nonoverlapping_CNN_all_tanh": lambda: models.NonOverlappingCNN(torch.tanh, torch.tanh, torch.tanh),
    "nonoverlapping_CNN_all_relu": lambda: models.NonOverlappingCNN(torch.relu, torch.relu, torch.relu),
    "nonoverlapping_CNN_all_sigmoid": lambda: models.NonOverlappingCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid),
    "overlapping_CNN_all_tanh": lambda: models.OverlappingCNN(torch.tanh, torch.tanh, torch.tanh),
    "overlapping_CNN_all_relu": lambda: models.OverlappingCNN(torch.relu, torch.relu, torch.relu),
    "overlapping_CNN_all_sigmoid": lambda: models.OverlappingCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid),
    "fcnn_all_tanh": lambda: models.FCNN(torch.tanh, torch.tanh, torch.tanh),
    "fcnn_all_relu": lambda: models.FCNN(torch.relu, torch.relu, torch.relu),
    "fcnn_all_sigmoid": lambda: models.FCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid)
}

def read_config():
    with open("config.json", "r") as file:
        config = json.load(file)   
    return config

def load_saved_models(save_dir, teacher_model, student_model): #this isn't used yet
    """Load the saved experiment data from a file."""
    
    # Construct the full path to the saved file
    save_file = f"{teacher_model}__{student_model}.pth"
    save_path = os.path.join(save_dir, save_file)

    # Load the data from the file
    checkpoint = torch.load(save_path)
    print(f"Loaded experiment from {save_path}")

    # Extract stored information
    teacher_model_state_dict = checkpoint["teacher_model_state_dict"]
    student_model_state_dict = checkpoint["student_model_state_dict"]
    final_loss = checkpoint["final_loss"]
    config = checkpoint["config"]

    return teacher_model_state_dict, student_model_state_dict, final_loss, config