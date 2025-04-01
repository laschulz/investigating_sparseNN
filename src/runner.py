import os
import torch
import torch.backends
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, teacher_model, student_model, teacher_name, student_name, lr, l1_norm, l2_norm, momentum=0.9, config_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on", self.device)
        
        self.config = utils.read_config(config_path)
        self.config_path = config_path

        # Initialize the teacher model with fixed ReLU activations
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model_name = teacher_name

        self.teacher_model = utils.init_teacher(self.teacher_model, self.teacher_model_name)
        self.teacher_model.to(self.device)

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
        self.optimizer = optim.SGD(self.student_model.parameters(), lr=self.lr, momentum=momentum)
        self.loss_fn = nn.MSELoss()
        # set loss to cosine similarity loss
        # self.loss_fn = nn.CosineEmbeddingLoss()

    def run(self):
        """Start the experiment: Generate dataset and train the student model."""
        # Generate dataset using the teacher model
        X_generated = torch.randn(self.config["dataset_size"], 12, device=self.device)
        with torch.no_grad():
            y_generated = self.teacher_model(X_generated).detach() #.cpu()
        X_generated = X_generated #.cpu()

        self.batch_size = self.config["batch_size"]
        self.clipping = self.config["clipping"]

        # Train student model
        self.student_model, self.final_loss = trainer.train_model(
            model=self.student_model,
            X_train=X_generated,
            y_train=y_generated,
            optimizer=self.optimizer,
            l1_lambda=self.l1_norm,
            l2_lambda=self.l2_norm,
            loss_fn=self.loss_fn,
            batch_size=self.batch_size,
            clipping=self.clipping,
            device=self.device,
            config_path=self.config_path
        )
    
    def evaluate(self):
        if "ViT" in self.student_model_name:
            # Perform evaluation using CKA metric for transformer models
            print(f"Evaluating transformer model: {self.student_model_name}")
            # Test on unseen data
            X_test = torch.randn(256, 12, device=self.device)
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
    
    def save_output(self):
        """Save the trained model's weights and log experiment details in a text file."""
        #Path
        date = datetime.now().strftime("%d%m%Y")
        name = self.config["name"]
        save_dir = os.path.join(self.config.get("save_path", "./experiment_output"), f"experiments_{date}_{name}")
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