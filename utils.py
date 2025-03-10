import json, os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

import models


model = {
    "nonoverlapping_CNN_all_tanh": lambda: models.nonoverlapping_CNN(torch.tanh, torch.tanh, torch.tanh),
    "nonoverlapping_CNN_all_relu": lambda: models.nonoverlapping_CNN(torch.relu, torch.relu, torch.relu),
    "nonoverlapping_CNN_all_sigmoid": lambda: models.nonoverlapping_CNN(torch.sigmoid, torch.sigmoid, torch.sigmoid)
}

def read_config():
    with open("config.json", "r") as file:
        config = json.load(file)   
    return config

def train_model(model, X_train, y_train, optimizer, loss_fn, l1_lambda=0, batch_size=32): #potentially add this to runner
    config = read_config()
    best_loss = float('inf')
    patience_counter = 0

    # Create a DataLoader to handle batching and shuffling
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training...")
    with tqdm(total=config["num_epochs"], desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(config["num_epochs"]):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y)

                # Apply L1 regularization if l1_lambda > 0
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss += l1_norm * l1_lambda

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()  # Accumulate batch loss

            # Compute average loss for the epoch
            epoch_loss /= len(dataloader)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')

            # Early stopping logic
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch}, best loss: {best_loss:.4f}")
                    break
    return model, best_loss

def load_saved_models(save_dir, teacher_model, student_model):
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

def calc_distance_metric(teacher_model, student_model, teacher_model_name: str, student_model_name: str):
    """ take (absolute) distance between each parameter of the teacher and student 
        (Note that these are the actual models and not the model names).
        Take absolute of a whole row if the activation function is symmetric at 0 -> tanh -> TODO: HOW TO CHECK THIS SMARTLY?
    """
    distance = 0.0
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        if "tanh" in teacher_model_name and "tanh" in student_model_name: #TODO: test if this is correct
            print("Entered tanh case") 
            distance += torch.norm(torch.abs(teacher_param) - torch.abs(student_param)).item()
        else:
            distance += torch.norm(teacher_param - student_param).item()
    return distance

def cka_metric(teacher_model, student_model):
    # TODO: IMPLEMENT, not yet relevant
    pass
