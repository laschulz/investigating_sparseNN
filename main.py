import sys, signal
import traceback, argparse
from functools import partial
import torch

from runner import ExperimentRunner
import utils
import models

def signal_handler(msg, sig, frame):
    """Handles timeout signals and logs experiment failures."""
    print(f"Exit signal received: {sig}")
    cmd, student_model_name = msg
    with open(f"timeout_{student_model_name}.txt", "a") as f:
        f.write(f"{cmd} \n")
    sys.exit(0)

def run_single_experiment(teacher_model, student_model, teacher_name, student_name, learning_rate):
    """Runs an experiment, trains the model, and saves results."""
    
    experiment_runner = ExperimentRunner(
        teacher_model=teacher_model,
        student_model=student_model,
        lr=learning_rate,
        teacher_name=teacher_name,
        student_name=student_name
    )

    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, student_name)))

    try:
        experiment_runner.run()
        experiment_runner.evaluate()
        experiment_runner.save_output()
    except Exception as e:
        with open(f"failed_{student_name}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run model experiments")
    parser.add_argument("--mode", type=str, choices=["single", "multiple"], required=True, help="Execution mode")
    parser.add_argument("--teacher_model", type=str, help="Teacher model name")
    parser.add_argument("--student_model", type=str, help="Student model name")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")

    args = parser.parse_args()
    print("Parsed arguments:", args)

    mode = args.mode
    learning_rate = args.lr

    if mode == "single":
        teacher_name = args.teacher_model
        student_name = args.student_model
        teacher_model = utils.model[teacher_name]()
        student_model = utils.model[student_name]()
        run_single_experiment(teacher_model, student_model, teacher_name, student_name, learning_rate)

    elif mode == "multiple":
        for t_act in {torch.tanh, torch.relu, torch.sigmoid}:
            for s_act in {torch.tanh, torch.relu, torch.sigmoid}:
                teacher_model = models.nonoverlapping_CNN(t_act, t_act, t_act)
                student_model = models.nonoverlapping_CNN(s_act, s_act, s_act) #TODO: replace this with generic model that is defined in args
                teacher_name = f"nonoverlapping_CNN_{t_act.__name__}"
                student_name = f"nonoverlapping_CNN_{s_act.__name__}"
                run_single_experiment(teacher_model, student_model, teacher_name, student_name, learning_rate)