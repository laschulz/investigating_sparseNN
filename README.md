# investigating_sparseNN

Repository for Semester Project conducted at the MIT Poggio Lab.


## Run Experiments

Currently, the code enables to either run a single experiment
```sh
python src/main.py \
    --mode single \
    --teacher_model multiWeight_CNN_all_tanh \
    --student_model fcnn_decreasing_all_tanh \
    --config_path config_2.json
```

or running multiple experiments where all 9 combinations of activation functions (tanh, sigmoid, relu) for the teacher / student are tested.
```sh
python src/main.py \
    --mode multiple \
    --teacher_type nonoverlappingCNN \
    --student_type nonoverlappingCNN \
    --config_path config_1.json
```

