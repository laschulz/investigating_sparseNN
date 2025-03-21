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

def gram_matrix(x):
    """Computes the Gram (covariance) matrix of input x."""
    return x @ x.T

def calc_cka_metric(teacher_model, student_model, data_loader, device="cpu", num_layers=3):
    """
    Computes the Centered Kernel Alignment (CKA) similarity between the CNN (teacher) 
    and ViT (student) models based on feature representations.
    """
    teacher_model.to(device).eval()
    student_model.to(device).eval()

    cka_scores = {}

    with torch.no_grad():
        for layer_index in range(num_layers):
            features_teacher = []
            features_student = []
            for batch in data_loader:
                x, _ = batch  # We don't need y_batch
                x = x.to(device)

                # Get features
                teacher_activations = extract_features(teacher_model, x, layer_index)
                features_teacher.append(teacher_activations)

                student_activations = extract_features(student_model, x, layer_index)
                features_student.append(student_activations)

            features_teacher = torch.cat(features_teacher, dim=0)
            features_student = torch.cat(features_student, dim=0)

            features_teacher = F.normalize(features_teacher, dim=1)
            features_student = F.normalize(features_student, dim=1)

            gram_teacher = gram_matrix(features_teacher).to(device)
            gram_student = gram_matrix(features_student).to(device)

            # Compute CKA similarity
            cka_score = torch.trace(gram_teacher @ gram_student) / (torch.norm(gram_teacher) * torch.norm(gram_student))
            cka_scores[layer_index] = cka_score.item()

    cka_sum = sum(cka_scores.values())
    return cka_sum

# TODO: this confuses me
def extract_features(model, x, layer_index):
    """
    Extracts feature representations from a given layer of the model.
    """
    activations = []

    def hook(module, input, output):
        activations.append(output)

    # Register hook to capture activations
    layers = list(model.children())
    handle = layers[layer_index].register_forward_hook(hook)

    with torch.no_grad():
        _ = model(x)

    handle.remove()  # Clean up hook

    return activations[0].reshape(x.shape[0], -1)  # Flatten features
