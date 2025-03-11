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

def model_mapper(model_type, activation):
    """Helper function to create the appropriate model instance."""
    model_map = {
        "nonoverlapping": models.nonoverlapping_CNN,
        "overlapping": models.overlapping_CNN,
        "fcnn": models.FCNN
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_map[model_type](activation, activation, activation)

def run_experiments(teacher_type, student_types):
    """Runs experiments across multiple activations for given teacher and student model types."""
    activations = {torch.tanh, torch.relu, torch.sigmoid}

    for t_act in activations:
        teacher_model = model_mapper(teacher_type, t_act)
        teacher_name = f"{teacher_type}_CNN_{t_act.__name__}"

        for s_act in activations:
            for student_type in student_types:
                student_model = model_mapper(student_type, s_act)
                student_name = f"{student_type}_CNN_{s_act.__name__}"
                run_single_experiment(teacher_model, student_model, teacher_name, student_name)

def run_single_experiment(teacher_model, student_model, teacher_name, student_name):
    """Runs an experiment, trains the model, and saves results."""
    
    experiment_runner = ExperimentRunner(
        teacher_model=teacher_model,
        student_model=student_model,
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
    parser.add_argument("--mode", type=str, choices=["single", "multiple", "all"], required=True, help="Execution mode")
    parser.add_argument("--teacher_model", type=str, help="Teacher model name")
    parser.add_argument("--student_model", type=str, help="Student model name")
    parser.add_argument("--student_type", type=str, choices=["nonoverlapping", "overlapping", "fcnn"],
                        help="Student model type: nonoverlapping, overlapping, fcnn")

    args = parser.parse_args()
    mode = args.mode
    print("Parsed arguments:", args)

    if mode == "single":
        teacher_name = args.teacher_model
        student_name = args.student_model
        teacher_model = utils.model[teacher_name]()
        student_model = utils.model[student_name]()
        run_single_experiment(teacher_model, student_model, teacher_name, student_name)

    elif mode == "multiple":
        if not args.student_type:
            raise ValueError("--student_type is required in 'multiple' mode")
        run_experiments("nonoverlapping", [args.student_type])

    elif mode == "all":
        run_experiments("nonoverlapping", ["nonoverlapping", "overlapping", "fcnn"])
