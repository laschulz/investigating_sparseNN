# investigating_sparseNN

Repository for Semester Project conducted at the MIT Poggio Lab.


## Run Experiments

Currently, the code enables to either run a single experiment
```sh
python3 main.py \
    --mode single \
    --teacher_model nonoverlapping_CNN_all_tanh \
    --student_model nonoverlapping_CNN_all_tanh \
    --lr 0.05
```

or running multiple experiments where all 9 combinations of activation functions (tanh, sigmoid, relu) for the teacher / student are tested.
```sh
python3 main.py \
--mode multiple \
--lr 0.05
```
