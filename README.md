# E1-246-Assignment3
CKY parser using PCFG

## About
This repository contains the implementation of cky algorithm for constituency parsing using probabilistic context free grammar (PCFG).

## Setup Instructions

### Setting up the environment
Python 3.6 is required to run the code in this repository. I have used python 3.6.7 

To install the requirements
```
pip3 install -r requirements.txt
```

### Dataset


### Setting the hyperparameters
All the hyperparameters are loaded from `config.json` file. The config file is assumed to be under the same directory as that of the python files. If the config file is to be loaded from different directory use `--config_file` to specify file path. 

Here I have briefly described each of these hyperparameter flags present in `config.json`.
* `smoothing` : the type of smoothing to use (takes values "backoff" or "interpolation")
* `alpha` : smoothing parameter
* `num_eval` : number of sentences to evaluate , required during test time

### How to run

To train the parser run:
```
python cky.py --mode train --data_dir data/
```
The `--data_dir` flag indicates the directory where the train, test data, grammar, probabilities and mappings file will be saved.

To parse an input sentence run:
```
python cky.py --mode parse --sentence John plays the guitar.
```

**Options**
1. Specify the data load dir using `--data_dir` if it is other that `data/`
2. Specify the log dir using `--log_dir` if it is other than `logs/`
3. Use the `-draw_tree` flag to draw the parsed tree for the input sentence

The parsed output is saved under `log_dir/` as `parsed_tree.tst`

To evaluate on the test set run:
```
python cky.py --mode test 
```

Options 1 and 2 are also applicable here.

## Pretrained Model
The `data` directory already contains a grammar, probabilities and mappings file.

## Evaluation
For evaluation I use the Evalb library [1] which is also uploaded in this repository.

## References
<cite>[1] Satoshi Sekine and Michael J. Collins. 1997. [Evalb](https://nlp.cs.nyu.edu/evalb/)</cite>
