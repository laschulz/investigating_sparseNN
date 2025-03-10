import os
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import utils


class ExperimentRunner:
    """
    Runs training or dataset generation based on command-line arguments.
    """

    # could change this that we have th emodel directly
    def __init__(self, teacher_model, student_model, teacher_name, student_name, lr=0.05, momentum=0.9):
        np.random.seed(42)
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on", self.device)

        # Load configuration
        self.config = utils.read_config()

        # Initialize the teacher model with fixed ReLU activations
        self.teacher_model = teacher_model.to(self.device)
        with torch.no_grad():
            self.teacher_model.conv1.weight.copy_(torch.tensor([[[2.59, -2.83, 0.87]]]))
            self.teacher_model.conv2.weight.copy_(torch.tensor([[[-1.38, 1.29]]]))
            self.teacher_model.conv3.weight.copy_(torch.tensor([[[0.86, -0.84]]]))

        # Initialize the student model
        self.student_model = student_model.to(self.device)

        # Define optimizer and loss function
        self.optimizer = optim.SGD(self.student_model.parameters(), lr=lr, momentum=momentum)
        self.loss_fn = nn.MSELoss()

        # Save model names for logging
        self.teacher_model_name = teacher_name
        self.student_model_name = student_name

    def evaluate(self):
        pass

    def run(self):
        """Start the experiment: Generate dataset and train the student model."""
        
        # Generate dataset using the teacher model
        X_generated = torch.randn(self.config["dataset_size"], 12).to(self.device)
        y_generated = self.teacher_model(X_generated).detach()

        # Train student model
        self.student_model, self.final_loss = utils.train_model(
            model=self.student_model,
            X_train=X_generated,
            y_train=y_generated,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            batch_size=self.config["batch_size"]
        )

        # TODO: maybe remove these print statements?
        print("\nTarget function parameters (Teacher Model):")
        for param in self.teacher_model.parameters():
            print(param.data.cpu().numpy())

        print("\nStudent function parameters AFTER training:")
        for param in self.student_model.parameters():
            print(param.data.cpu().numpy())

    def save_output(self):
        """Save the trained model's weights."""
        save_data = {
            "teacher_model_state_dict": self.teacher_model.state_dict(),
            "student_model_state_dict": self.student_model.state_dict(),
            "final_loss": self.final_loss,
            "config": self.config
            # TODO: save activations when this becomes relevant
        }
        save_dir = self.config.get("save_path", "./experiment_output")
        save_file = f"{self.teacher_model_name}__{self.student_model_name}.pth"
        save_path = os.path.join(save_dir, save_file)
        torch.save(save_data, save_path)
        print(f"Experiment saved to {save_path}")