import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def calc_distance_metric(teacher_model, student_model, teacher_model_name: str, student_model_name: str, device='cpu'):
    """ 
    Take (absolute) distance between each parameter of the teacher and student 
    (Note that these are the actual models and not the model names).
    Take absolute of a whole row if the activation function is symmetric at 0 -> tanh
    """
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    total_distance = 0.0
    use_absolute_distance = "tanh" in teacher_model_name and "tanh" in student_model_name

    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param = t_param.to(device)
        s_param = s_param.to(device)
    
        # If using tanh, take absolute values before matching
        if use_absolute_distance:
            t_param = torch.abs(t_param)
            s_param = torch.abs(s_param)

        total_distance += match_layer_weights(teacher_weights=t_param, student_weights=s_param, device=device)

    return total_distance


def match_layer_weights(teacher_weights: torch.Tensor, student_weights: torch.Tensor, threshold=1e-4, device='cpu'):
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
    cost_matrix = torch.cdist(t_weights.unsqueeze(1), s_weights.unsqueeze(1), p=2).to(device)  # Shape: (num_teacher, num_student)
    teacher_indices, student_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    matched_distance = torch.norm(t_weights[teacher_indices] - s_weights[student_indices]).item()

    # Handle extra student weights (unmatched)
    unmatched_students = set(range(s_weights.shape[0])) - set(student_indices)
    unmatched_penalty = torch.norm(s_weights[list(unmatched_students)]).item() if unmatched_students else 0.0

    total_distance = matched_distance + unmatched_penalty
    return total_distance
