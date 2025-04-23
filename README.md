# investigating_sparseNN


This repository contains the code for the project [L1-regularization: Path to sparsity](./l1_path_to_sparsity.pdf), conducted at the MIT Poggio Lab. 

This study investigates the effects of L1-regularization on inducing sparsity in neural networks, specifically within a teacher-student framework. Various experiments were conducted to evaluate both the convergence and emergence of sparsity. 
The results show that L1-regularization is the only regularization technique that reliably drives weights exactly to zero. It achieves near-perfect sparsity, especially when the teacher and student share the same activation function. While initialization is crucial, our results offer practical guidance for optimizing student networks that accurately approximate teachers and recover sparse, interpretable representations.

### Abstract

---

## Project Structure

The repository is organized as follows:

- **`src/`**: Contains the main source code for running experiments, including model definitions, training loops, and evaluation scripts.
- **`scripts/`**: Includes scripts to run each experiment on the cluster.
- **`config_files/`**: Stores JSON configuration files that define hyperparameters and settings for the experiments.
- **`experiment_output/`**: Directory that contains all results from the experiments, including model checkpoints and brief insights into the weights, loss variables, etc in a text-file.
- **`analysis/`**: Contains Jupyter notebooks for each experiment for analyzing experiment results, visualizing sparsity patterns.
- **`requirements.txt`**: Lists the Python dependencies required to run the project.

---

## Running Experiments

### Single Experiment

To run a specific teacher–student experiment with a defined configuration:

```sh
python src/main.py \
    --mode single \
    --teacher_model baselineCNN_tanh \
    --student_model fcn_256_32_relu \
    --config_path config_noReg_0.2.json \
    --seed 5 \
    --name exp3
```

In this mode, `--teacher_model` and `--student_model` should be specified using the format:
<model_type>_<activation_function>

where:
- `<model_type>` $\in$ `{baselineCNN, multiChannelCNN, splitFilterCNN, fcn_128_128, fcn_256_32}`
- `<activation_function>` $\in$ `{relu, sigmoid, tanh}`


### Multiple Experiments

To run multiple experiments over combinations of activation functions for a given teacher and student model type:

```sh
python src/main.py \
    --mode multiple \
    --teacher_type baselineCNN \
    --student_type fcn_128_128 \
    --config_path config.json \
    --seed 42 \
    --name my_experiments
```

**Note:** The arguments `--seed` and `--name` are optional and default to `42` and `"noName"` respecively. 
All other hyperparameters, such as learning rate, regularization strength, and activation constraints, are defined in the corresponding configuration file.

> **Note:** Only the name of the config file (e.g., `config.json`) needs to be passed.  
> It is assumed to be located in the `config_files/` folder.
