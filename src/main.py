import sys, signal
import traceback, argparse
from functools import partial
import torch
import itertools
import numpy as np

from runner import ExperimentRunner
import utils
import models

def signal_handler(msg, sig):
    """Handles timeout signals and logs experiment failures."""
    print(f"Exit signal received: {sig}")
    cmd, student_model_name = msg
    with open(f"timeout_{student_model_name}.txt", "a") as f:
        f.write(f"{cmd} \n")
    sys.exit(0)

def model_mapper(model_type, activation, config_path):
    """Helper function to create the appropriate model instance."""
    model_map = {
        "baselineCNN": models.BaselineCNN,
        "splitFilterCNN": models.SplitFilterCNN,
        "multiChannelCNN": models.MultiChannelCNN,
        "fcn_128_128": models.FCN_128_128,
        "fcn_256_32": models.FCN_256_32,
        "fcn_1024_128": models.FCN_1024_128
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    if model_type == "nonoverlappingViT":
        return model_map[model_type](activation, config_path)
    else:
        return model_map[model_type](activation, activation, activation, config_path)

def run_experiments(teacher_type, student_types, config_path, seed, name = "NoName"):
    """Runs experiments across multiple activations for given teacher and student model types."""
    activations = {torch.sigmoid, torch.tanh, torch.relu}

    config = utils.read_config(config_path)
    lr_values = config.get("lr", [0.01])
    l1_norm_values = config.get("l1_norm", [0])
    l2_norm_values = config.get("l2_norm", [0])

    # Generate all possible hyperparameter combinations
    param_combinations = list(itertools.product(
        student_types,
        activations,
        activations,
        lr_values,
        l1_norm_values,
        l2_norm_values
    ))

    for student_type, teacher_activation, student_activation, lr, l1_norm, l2_norm in param_combinations:
        if teacher_activation != student_activation and config["same_act"]:
            continue
        teacher_model = model_mapper(teacher_type, teacher_activation, config_path)
        teacher_name = f"{teacher_type}_{teacher_activation.__name__}"

        student_model = model_mapper(student_type, student_activation, config_path)
        student_name = f"{student_type}_{student_activation.__name__}"

        print(f"Running experiment with: "
              f"Teacher={teacher_type}, Student={student_type}, "
              f"t_act={teacher_activation.__name__}, s_act={student_activation.__name__}, "
              f"lr={lr}, l1_norm={l1_norm}, l2_norm={l2_norm}")

        run_single_experiment(
            teacher_model=teacher_model,
            student_model=student_model,
            teacher_name=teacher_name,
            student_name=student_name,
            lr=lr,
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            config_path=config_path,
            seed=seed,
            name=name
        )

def run_single_experiment(teacher_model, student_model, teacher_name, student_name, lr, l1_norm, l2_norm, config_path, seed, name = "NoName"):
    """Runs an experiment, trains the model, and saves results."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    teacher_model = teacher_model
    student_model = student_model
    
    experiment_runner = ExperimentRunner(
        teacher_model=teacher_model,
        student_model=student_model,
        teacher_name=teacher_name,
        student_name=student_name,
        lr=lr,
        l1_norm=l1_norm,
        l2_norm=l2_norm,
        config_path=config_path,
        seed=seed,
        name=name
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
    parser.add_argument("--teacher_type", type=str, choices=["baselineCNN", "splitFilterCNN", "multiChannelCNN"],
                        help="Student model type: baselineCNN, splitFilterCNN, multiChannelCNN")
    parser.add_argument("--student_type", type=str, choices=["baselineCNN", "multiChannelCNN", "fcn_128_128", "fcn_256_32"],
                        help="Student model type: baselineCNN, multiChannelCNN, fcn_128_128, fcn_256_32")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--name", type=str, help="Name of the experiment")

    args = parser.parse_args()
    mode = args.mode

    if mode == "single":
        teacher_name = args.teacher_model
        student_name = args.student_model
        teacher_model = utils.create_model(teacher_name, args.config_path)
        student_model = utils.create_model(student_name, args.config_path)
        config = utils.read_config(args.config_path)
        lr = config["lr"][0]
        l1_norm = config["l1_norm"][0]
        l2_norm = config["l2_norm"][0]

        run_single_experiment(teacher_model=teacher_model, 
                              student_model=student_model, 
                              teacher_name=teacher_name, 
                              student_name=student_name, 
                              lr=lr, 
                              l1_norm=l1_norm, 
                              l2_norm=l2_norm,
                              config_path=args.config_path,
                              seed=args.seed,
                              name=args.name
                              )

    elif mode == "multiple":
        if not args.student_type:
            raise ValueError("--student_type is required in 'multiple' mode")
        run_experiments(args.teacher_type, [args.student_type], config_path=args.config_path, seed=args.seed, name=args.name)