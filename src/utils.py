import json, os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import models

model = {
    "nonoverlapping_CNN_all_tanh": lambda: models.nonoverlapping_CNN(torch.tanh, torch.tanh, torch.tanh),
    "nonoverlapping_CNN_all_relu": lambda: models.nonoverlapping_CNN(torch.relu, torch.relu, torch.relu),
    "nonoverlapping_CNN_all_sigmoid": lambda: models.nonoverlapping_CNN(torch.sigmoid, torch.sigmoid, torch.sigmoid),
    "overlapping_CNN_all_tanh": lambda: models.overlapping_CNN(torch.tanh, torch.tanh, torch.tanh),
    "overlapping_CNN_all_relu": lambda: models.overlapping_CNN(torch.relu, torch.relu, torch.relu),
    "overlapping_CNN_all_sigmoid": lambda: models.overlapping_CNN(torch.sigmoid, torch.sigmoid, torch.sigmoid),
    "fcnn_all_tanh": lambda: models.FCNN(torch.tanh, torch.tanh, torch.tanh),
    "fcnn_all_relu": lambda: models.FCNN(torch.relu, torch.relu, torch.relu),
    "fcnn_all_sigmoid": lambda: models.FCNN(torch.sigmoid, torch.sigmoid, torch.sigmoid)
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

def calc_distance_metric(teacher_model, student_model, teacher_model_name: str, student_model_name: str):
    """ take (absolute) distance between each parameter of the teacher and student 
        (Note that these are the actual models and not the model names).
        Take absolute of a whole row if the activation function is symmetric at 0 -> tanh
    """
    total_distance = 0.0
    use_absolute_distance = "tanh" in teacher_model_name and "tanh" in student_model_name

    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        # If using tanh, take absolute values before matching
        if use_absolute_distance:
            t_param = torch.abs(t_param)
            s_param = torch.abs(s_param)

        total_distance += match_layer_weights(t_param, s_param)

    return total_distance


def match_layer_weights(teacher_weights: torch.Tensor, student_weights: torch.Tensor, threshold=1e-4):
    """
    Matches the closest student weights to teacher weights per layer.
    - Uses the Hungarian algorithm to find the optimal one-to-one matching using Euclidian distance.
    - If a student weight is below `threshold`, it's treated as **zero** (sparsity assumption).
    - Returns the matched distance and a penalty for extra student weights.
    """

    t_weights = teacher_weights.view(-1)  
    s_weights = student_weights.view(-1)  

    # Apply threshold: Consider near-zero student weights as 0 (sparsity assumption)
    s_weights = torch.where(torch.abs(s_weights) < threshold, torch.tensor(0.0, device=s_weights.device), s_weights)

    # Find optimal 1-to-1 matching
    cost_matrix = torch.cdist(t_weights.unsqueeze(1), s_weights.unsqueeze(1), p=2)  # Shape: (num_teacher, num_student)
    teacher_indices, student_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    matched_distance = torch.norm(t_weights[teacher_indices] - s_weights[student_indices]).item()

    # Handle extra student weights (unmatched)
    unmatched_students = set(range(s_weights.shape[0])) - set(student_indices)
    unmatched_penalty = torch.norm(s_weights[list(unmatched_students)]).item() if unmatched_students else 0.0

    total_distance = matched_distance + unmatched_penalty
    return total_distance

def cka_metric(teacher_model, student_model):
    # TODO: IMPLEMENT, not yet relevant
    pass