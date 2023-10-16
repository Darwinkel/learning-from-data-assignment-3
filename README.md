# Learning from Data - Assignment 3

## Requirements
See `requirements.txt` and `requirements-dev.txt`. The code targets Python 3.10 and 3.11.


## How to run
If you want to replicate our experiment, including performing the hyperparameter search and running the best model on the test set, you can use `make lstm` and `make lm`. 

If you want to customize the experiment (e.g. change the configuration and hyperparameters of the models), see the output of:

`python lfd_assignment3_lstm.py --help`
and
`python lfd_assignment3_lm.py --help`

The files `jobscript_lstm.sh` and `jobscript_lm.sh` can be used to setup a working environment on Hábrók (usable with `sbatch`).
Note that a virtual environment should be created beforehand. 

## Data
The used datasets are in the `data` folder. They were generated from the supplied assignment data, by randomly shuffling the texts with the Unix utility `shuf`, before we split the data into a train set (80\%), a development set (10\%), and a test set (10\%).

## Results
The output from Hábrók's jobscripts can be found in the `results` folder. 