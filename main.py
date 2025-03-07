import sys
import os
import time
import signal
import traceback
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np

import utils
import models


class ExperimentRunner:
    """
    Runs training or dataset generation based on command-line arguments.
    """

    def __init__(self, teacher_model, student_model, lr = 0.05, momentum = 0.9):
        np.random.seed(42)
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on", self.device)

        # Load configuration
        self.config = utils.read_config()

        # Initialize the model
        self.teacher_model = utils.model[teacher_model]().to(self.device)
        with torch.no_grad():
            self.teacher_model.conv1.weight.copy_(torch.tensor([[[2.59, -2.83, 0.87]]]))
            self.teacher_model.conv2.weight.copy_(torch.tensor([[[-1.38, 1.29]]]))
            self.teacher_model.conv3.weight.copy_(torch.tensor([[[0.86, -0.84]]]))
        self.student_model = utils.model[student_model]().to(self.device)

        self.optimizer = optim.SGD(self.student_model.parameters(), lr=lr, momentum=momentum)
        self.loss_fn = torch.nn.MSELoss()

    def start(self):
        """Start the selected experiment."""
        
        X_generated, y_generated = utils.generate_dataset()
        self.student_model, self.final_loss = utils.train_model(model=self.student_model, X_train=X_generated, y_train=y_generated, 
                          optimizer=self.optimizer, loss_fn=self.loss_fn, batch_size=self.config["batch_size"])
        
        print("\nTarget function parameters:")
        for param in self.teacher_model.parameters():
            print(param.data.numpy())
        
        print("\nStudent function parameters AFTER training:")
        for param in self.student_model.parameters():
            print(param.data.numpy())

    def save_output(self):
        """Save the trained model's weights."""
        save_data = {
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "final_loss": self.final_loss,
            "config": self.config
        }
        save_path = self.config["save_path"]
        torch.save(save_data, save_path)
        print(f"Experiment saved to {save_path}")


def signal_handler(msg, signal, frame):
    """Handles timeout signals and logs experiment failures."""
    print(f"Exit signal received: {signal}")
    cmd, model = msg
    with open(f"timeout_{model}.txt", "a") as f:
        f.write(f"{cmd} \n")
    sys.exit(0)


if __name__ == "__main__":
    # Parse command-line arguments
    args_list = sys.argv[1:]
    args = {k[2:]: v for k, v in zip(args_list[::2], args_list[1::2])}

    if "model" not in args:
        print("Usage: python main.py --teacher_model <model_name> --student_model <model_name> --lr") #have a range?
        sys.exit(1)

    # Instantiate and run experiment
    experiment_runner = ExperimentRunner(args["teacher_model"], args["student_model", args["lr"]])
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args["model"])))

    # Track execution time
    start_time = time.time()
    try:
        experiment_runner.start()
        with open(f"finished_{args['model']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time() - start_time} \n")
    except Exception as e:
        with open(f"failed_{args['model']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['model']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")
