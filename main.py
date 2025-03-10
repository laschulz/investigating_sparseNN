import sys, signal, time
import traceback
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

    start_time = time.time()
    try:
        experiment_runner.run()
        experiment_runner.evaluate()
        experiment_runner.save_output()
        with open(f"finished_{student_name}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time() - start_time:.2f} seconds\n")
    except Exception as e:
        with open(f"failed_{student_name}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")


if __name__ == "__main__":
    # Parse command-line arguments
    args_list = sys.argv[1:]
    args = {k[2:]: v for k, v in zip(args_list[::2], args_list[1::2])}
    print("Parsed arguments:", args)  # Debug print statement

    mode = args.get("mode")
    learning_rate = float(args.get("lr", 0.05))

    # Ensure required arguments exist
    # if "teacher_model" not in args or "student_model" not in args:
    #     print("Usage: python main.py --teacher_model <model_name> --student_model <model_name> --lr <learning_rate>")
    #     sys.exit(1)

    if mode == "single":
        teacher_model = utils.model[args["teacher_model"]]()
        student_model = utils.model[args["student_model"]]()
        teacher_name = args["teacher_model"]
        student_name = args["student_model"]
        run_single_experiment(teacher_model, student_model, teacher_name, student_name, learning_rate)

    elif mode == "multiple":
        for t_act in {torch.tanh, torch.relu, torch.sigmoid}:
            for s_act in {torch.tanh, torch.relu, torch.sigmoid}:
                teacher_model = models.nonoverlapping_CNN(t_act, t_act, t_act)
                student_model = models.nonoverlapping_CNN(s_act, s_act, s_act) #TODO: replace this with generic model that is defined in args
                teacher_name = f"nonoverlapping_CNN_{t_act}"
                student_name = f"nonoverlapping_CNN_{s_act}"
                run_single_experiment(teacher_model, student_model, teacher_name, student_name, learning_rate)