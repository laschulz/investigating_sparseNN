import os
import torch
import torch.backends
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

import utils
import metrics
import trainer

class ExperimentRunner:
    """
    Runs training or dataset generation based on command-line arguments.
    """

    # could change this that we have th emodel directly
    def __init__(self, teacher_model, student_model, teacher_name, student_name, lr, l1_norm, l2_norm, momentum=0.9, device="cpu", config_path=None):
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.config = utils.read_config(config_path)
        self.device = device
        self.config_path = config_path
        print("Running on", self.device)

        # Initialize the teacher model with fixed ReLU activations
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model_name = teacher_name
        if "nonoverlapping" in teacher_name:
            teacher_weights = [
                torch.tensor([[[2.59, -2.83, 0.87]]], device=self.device),  # conv1
                torch.tensor([[[-1.38, 1.29]]], device=self.device),        # conv2
                torch.tensor([[[0.86, -0.84]]], device=self.device)         # conv3
            ]
            with torch.no_grad():
                for layer, weight in zip(self.teacher_model.layers, teacher_weights):
                    layer.weight.copy_(weight)
        elif "overlapping" in teacher_name:
            teacher_weights = [
            torch.tensor([[[-0.78, -0.12,  0.70]],
                      [[-1.16,  0.47,  0.05]],
                      [[-0.73,  1.96, -1.01]],
                      [[-0.32,  0.21,  0.63]]], device=self.device),  # conv1
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
                       [ 0.15,  0.49]]], device=self.device),  # conv2
            torch.tensor([[[-0.54,  0.16],
                       [-0.74, -0.46],
                       [ 0.08,  0.18],
                       [-0.22,  0.81]]], device=self.device)           # conv3
            ]
    
            with torch.no_grad():
                for layer, weight in zip(self.teacher_model.layers, teacher_weights):
                    layer.weight.copy_(weight)

        # print weights of model
        for name, param in self.teacher_model.named_parameters():
            print(name, param.data)

        # Initialize the student model
        self.student_model = student_model.to(self.device)
        self.student_model_name = student_name

        # Define optimizer and loss function
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm
        self.lr = lr
        self.optimizer = optim.SGD(self.student_model.parameters(), lr=self.lr, momentum=momentum, weight_decay=self.l2_norm)
        self.loss_fn = nn.MSELoss()

    def evaluate(self):
        if "ViT" in self.student_model_name:
            # Perform evaluation using CKA metric for transformer models
            print(f"Evaluating transformer model: {self.student_model_name}")
            # Test on unseen data
            X_test = torch.randn(256, 12).to(self.device)
            y_test = self.teacher_model(X_test).detach().to(self.device)
            dataset = TensorDataset(X_test, y_test)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.distance = metrics.calc_cka_metric(teacher_model=self.teacher_model, student_model=self.student_model, data_loader=dataloader, device=self.device)
        else:
            # Perform general evaluation using the normal distance metric
            print(f"Evaluating model: {self.student_model_name}")
            self.distance = metrics.calc_distance_metric(teacher_model=self.teacher_model, 
                                                         student_model=self.student_model, 
                                                         teacher_model_name=self.teacher_model_name, 
                                                         student_model_name=self.student_model_name, 
                                                         device=self.device)
    
    def run(self):
        """Start the experiment: Generate dataset and train the student model."""
        # Generate dataset using the teacher model
        # X_generated = torch.randn(self.config["dataset_size"], 12, device=self.device)
        # y_generated = self.teacher_model(X_generated).detach()
        X_generated = torch.randn(self.config["dataset_size"], 12)
        y_generated = self.teacher_model(X_generated).detach()

        self.batch_size = self.config["batch_size"]
        self.clipping = self.config["clipping"]

        # Train student model
        self.student_model, self.final_loss = trainer.train_model(
            model=self.student_model,
            X_train=X_generated,
            y_train=y_generated,
            optimizer=self.optimizer,
            l1_lambda=self.l1_norm,
            loss_fn=self.loss_fn,
            batch_size=self.batch_size,
            clipping=self.clipping,
            device=self.device,
            config_path=self.config_path
        )
    
    def save_output(self):
        """Save the trained model's weights and log experiment details in a text file."""
        #Path
        date = datetime.now().strftime("%d%m%Y")
        save_dir = os.path.join(self.config.get("save_path", "./experiment_output"), f"experiments_{date}")
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"{self.teacher_model_name}__{self.student_model_name}.pth")
        text_save_path = os.path.join(save_dir, f"experiment__{date}.txt")

        save_data = {
            "teacher_model_state_dict": self.teacher_model.state_dict(),
            "student_model_state_dict": self.student_model.state_dict(),
            "teacher_model_name": self.teacher_model_name,
            "student_model_name": self.student_model_name,
            "l1_norm": self.l1_norm,
            "l2_norm": self.l2_norm,
            "lr": self.lr,
            "final_loss": self.final_loss,
            "distance_metric": self.distance,
            "config": self.config
        }
        torch.save(save_data, model_save_path)
        print(f"Experiment saved to {model_save_path}")

        # Check if file already exists
        file_exists = os.path.exists(text_save_path)

        # Save experiment details in a text file
        with open(text_save_path, "a") as f:
            if not file_exists:
                f.write("Experiment Summary:\n")
                f.write("=" * 80 + "\n\n")

                # Save Teacher Model Parameters
                f.write("Teacher Model Parameters:\n")
                for name, param in self.teacher_model.named_parameters():
                    f.write(f"{name}: {param.data.cpu().numpy()}\n")
                f.write("\n" + "=" * 80 + "\n\n")

            f.write(f"{self.teacher_model_name} -> {self.student_model_name}\n\n")
            f.write("Student Model Parameters:\n")
            threshold = 1e-4 #was 1e-2
            for name, param in self.student_model.named_parameters():
                param_data = param.data.cpu().numpy()
                param_data[abs(param_data) < threshold] = 0  # Zero out small values
                f.write(f"{name}: {param_data}\n")

            f.write(f"\nFinal Loss: {self.final_loss:.4f}\n")
            f.write(f"Distance Metric: {self.distance:.4f}\n")
            f.write(f"L1 norm: {self.l1_norm}\n")
            f.write(f"L2 norm: {self.l2_norm}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Clipping: {self.clipping}\n")
            f.write(f"Learning rate: {self.lr}\n")
            f.write(f"data size: {self.config['dataset_size']}\n")
            f.write("\n" + "=" * 80 + "\n\n")

        print(f"Experiment details saved to {text_save_path}")
