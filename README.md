# investigating_sparseNN

Repository for Semester Project conducted at the MIT Poggio Lab.


## Run Experiments

Currently, the code enables to either run a single experiment
```sh
python src/main.py \
    --mode single \
    --teacher_model nonoverlapping_CNN_all_tanh \
    --student_model nonoverlapping_CNN_all_tanh \
    --config_path config_b16.json
```

or running multiple experiments where all 9 combinations of activation functions (tanh, sigmoid, relu) for the teacher / student are tested.
```sh
python src/main.py \
    --mode multiple \
    --student_type overlapping \
    --config_path config_b16.json
```
