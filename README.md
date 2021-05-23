# NodePert: A Perturbation Based Method for Training Deep Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

[**Setup**](#setup)
| [**Running NodePert**](#running-nodepert)
| [**Paper Experiments**](figs/running-paper-exps.md)

## Introduction
What algorithms drive goal directed learning in networks in the brain? In machine learning, networks are almost exclusively trained using stochastic gradient descent, which delivers errors tailored to each neuron in a network. However, computing such tailored errors requires complicated machinery that is unlikely to exist in the brain, at least not in all areas. An alternative is to apply random perturbations to parameters, see whether that increases or decreases a global error, and then adjust parameters accordingly. The fact that this approach does not require complicated machinery, along with evidence that the brain utilizes global signals for learning, has prompted neuroscientists to speculate that the brain may use perturbation algorithms. However, little is known about the efficacy of these algorithms for training large networks to perform complex tasks. This repository contains the code for performing a thorough empirical investigation of a fast perturbation method, and see how it scales on large and convolutional network architectures and datasets for image classification tasks. Our results provide insights into the diverse credit assignment algorithms employed by the brain.

The code was written using [JAX](https://github.com/google/jax) and TFDS (Tensorflow Datasets) was used for the dataloaders.

## Setup

Pull the repository from git via:

```bash
git clone https://github.com/silverpaths/nodepert.git
```

See requirements.txt file for the project dependencies, which can be installed via pip. Note, by default, this installs JAX with only CPU support. For installing JAX with GPU support, uncomment the line in requirements.txt.

```bash
pip install -r requirements.txt
```
To run a basic test to check that the JAX setup is alright, change into the cloned directory, open a Python
interpreter and issue:

```python
export PYTHONPATH <path/to/nodepert>
import fc_test
```
Check out the their official [install guide](https://github.com/google/jax#installation), for any JAX installation related issues.

## Running NodePert

#### Running on a single GPU/CPU setup
To train a basic network with node perturbation on MNIST, run:
>Fully Connected Network (FC network architecture parameterized by the arguments passed, i.e. runs network with 2 hidden layers, 500 neurons in each layer)
>```python
>export PYTHONPATH <path/to/nodepert>
>python fc_test.py -log_expdata True -n_hl 2 -hl_size 500 -lr 5e-3 -batchsize 100 -num_epochs 10 -update_rule np
>```
>Small Convolution Network (Architecture fixed at 4 conv layers, 32 channels in each layer)
>```python
>export PYTHONPATH <path/to/nodepert>
>python small_conv_test.py -log_expdata True -batchsize 100 -num_epochs 10 -update_rule sgd
>```
Check out the _utils.py_ file for the different arguments which may be passed along with their default values. The argument ```job_id``` is for running the same code with different parameters in parallel on a cluster.

#### Running on a compute cluster with SLURM

The slurm_scripts folder contains the scripts to run experiments on a multinode setup with the SLURM resource manager. The default ```job_id``` is 0, which runs the code with the parameter specified in the 0 <sup>th</sup> index of the list named _rows_ in the code. Make use of the slurm scripts, run:

```bash
bash slurm-scripts/meta_jobscript.sh fc-test
```
#### Running Experiments from the Paper: [check here](figs/running-paper-exps.md)

## Code Structure
The entire code runs on MNIST by default, unless the data_loader is explicitly changed in the ```npimports.py``` file. The experiments folder contains the code for all the experiments for the paper 'On the Limitations of Perturbation Based Methods for Training Deep Networks'. On your local machine (with GPU or CPU, and the default setup is a fully connected network), run:

```python
python experiments/vary_lr.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 100 -update_rule np
```
For creating a neural network, all files call the same ```models/conv.py``` or ```models/fc.py```. The function for the exact nodepert update can be found in ```models/optim.py```.

## License
The source code for this project is licensed under the [MIT license](LICENSE.md).
