# investigating_sparseNN


This repository contains the code for the project [L1-regularization: Path to sparsity](./l1_path_to_sparsity.pdf), conducted at the MIT Poggio Lab. 
This project investigates how student neural networks can learn sparse internal representations from fixed teacher networks. 
The core focus is on evaluating the effects of regularization techniques (L1-Regularization, L2-Regularization, no Regularization), and weight initialization on sparsity across various activation functions.

---

## Project Structure

TODO

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



python src/main.py \
    --mode single \
    --teacher_model baselineCNN_relu \
    --student_model fcn_128_128_tanh \
    --config_path config_l1_0.2.json \
    --seed 2 \
    --name exp4_l1_0.2
