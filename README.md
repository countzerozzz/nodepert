# 🧠 NodePert: Perturbation-Based Algorithms for Training Deep Neural Networks

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python)](https://docs.python.org/3/whatsnew/3.11.html)
[![JAX](https://img.shields.io/badge/Framework-JAX-important?style=for-the-badge&logo=Apache-Kafka)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative)](https://github.com/countzerozzz/nodepert/edit/master/LICENSE.md)

[**Setup**](#setup)
| [**Running NodePert**](#running-node-perturbation)
| [**Paper Experiments**](figs/running-paper-exps.md)

## Overview
What algorithms underlie goal directed learning in the brain? Backpropagation is the standard **credit assignment algorithm** used in machine learning research, but it's considered biologically implausible. Recently, **biologically plausible** alternatives, such as feedback alignment, target propagation, and perturbation algorithms, have been explored. The node perturbation algorithm applies random perturbations to neuron activity, monitors performance, and adjusts weights accordingly. This approach is simple and may be utilized by the brain. 

This repository contains the accompanying code for the paper, *An empirical study of perturbation methods for training deep networks*. It offers a *efficient* and *scalable* implementation of perturbation algorithms, allowing for large-scale experiments with node perturbation on **modern convolutional architectures on a GPU**. Our results provide insights into the diverse credit assignment algorithms used by the brain. The code was written by [Yash Mehta](https://yashsmehta.github.io/) and [Timothy Lillicrap](https://contrastiveconvergence.net/~timothylillicrap/index.php) using [**`JAX`**](https://github.com/google/jax) in conjunction with `Tensorflow Datasets` for data loading. Reach out to yashsmehta95@gmail.com or timothy.lillicrap@gmail.com with queries or feedback.

## Setup

1. Clone the repository like the git wizard you know you are:
    ```bash
    git clone https://github.com/silverpaths/nodepert.git
    cd nodepert
    ```

2. Create a new virtual environment using `venv` or `conda`. Note, `venv` comes inbuilt with python but we recommend using `conda`, especially if you want to run it on a GPU.

    <details>
    <summary> conda </summary>
    ```bash
    conda create -n nodepert python=3.11
    conda activate nodepert
    ```
    </details>


    <details>
    <summary> venv </summary>
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    </details>

3. Install JAX and the nodepert package:

    a. **CPU only**
    ```bash
    pip install --upgrade "jax[cpu]"
    pip install -e .
    ```

    b. **GPU**
    JAX currently only supports accelerated GPU computing on linux-based systems. Try "jax[cuda11_pip]" if cuda12_pip does not work out-of-the-box.
    
    <details>
    <summary> conda </summary>

    ```bash
    conda install -c nvidia cuda-toolkit
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -e .
    ```

    </details>


    <details>
    <summary> venv </summary>
    ```bash
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -e .
    ```
    </details>
    

3. To ensure JAX is working properly, run a basic experiment on a fully connected network comparing NP and SGD on MNIST. This should save a trajectory figure, while taking less than 2m to run.

    ```python
    python run_example.py
    ```

Run into any JAX installation snafus? Check out their [**official install guide**](https://github.com/google/jax#installation) for a helping hand.

## Running Node Perturbation

You can pass different arguments pertaining to the experiment, such as the number of hidden layers `n_hl`, hidden layer size `hl_size`, learning rate `lr`, logging training data `log_expdata`, etc. Check the entire list of parameters, along with their default values in `utils.py` `parse_args()` or `parse_conv_args()`. By default, the code runs on MNIST, which can be changed to f-MNIST, CIFAR10, CIFAR100 by changing the dataloader in the `npimports.py`.

#### Basic runs
>Here is a sample script for a Fully Connected Network (FC network architecture with 2 hidden layers and 500 neurons each):
>```python
>python fc_test.py -log_expdata True -n_hl 2 -hl_size 500 -lr 5e-3 -batchsize 100 -num_epochs 10 -update_rule np
>```
>For a Small Convolution Network (fixed small architecture of 4 conv layers, 32 channels each. To modify the small conv architecture, directly make changes within `small_conv_test.py`):
>```python
>python small_conv_test.py -log_expdata True -batchsize 100 -num_epochs 10 -update_rule sgd
>```
>For a Large Convolution Network:
>```python 
>python experiments/conv/all_cnn_net.py
>```

#### Detailed experiments

Inside the experiments folder, you'll find code and in-depth analysis of various experiments utilizing node perturbation. These experiments can be classified into the following types:

1. **Scaling**. See `dataset.py`, `depth.py`, `width.py`, `batchsize.py`, `learning_rate.py`
2. **Understanding network crashes during training**. See `crash-dynamics.py`, `crash_timing.py`, `width.py`, `batchsize.py`, `grad_dynamics.py`
3. **Relative change in the loss with different learning rates**. See `linesearch.py`, `linesearch_utils.py`
4. **Adam-like update for NP gradients**. See `adam_update.py`
5. **Visualizing the loss landscape**. See `loss_landscape.py`

And for all you neural network aficionados, take a gander at ```models/conv.py``` or ```models/fc.py```. The exact nodepert update can be found in ```models/optim.py```.

#### Running on a compute cluster
To maximize your resources and make the most of your multinode setup during experiments, consider using the job_id argument. It allows you to run multiple configurations and seeds of your experiments simultaneously, which is especially useful when working with GPU clusters that have resource allocation managers like SLURM.
```bash
bash slurm-scripts/meta_jobscript.sh fc-test
```

[**Running Experiments from the Paper**](figs/running-paper-exps.md)

## Citing
If you use this code in your own work, please use the following bibtex entry:

```bibtex
@misc{nodepert-2023,
  title={NodePert: An empircal study of perturbation methods for training deep networks}, 
  author={Mehta, Yash and Hiratani, Naoki and Humphreys, Peter and Latham, Peter and Lillicrap, Timothy}, 
  year={2023}, publisher={GitHub},
  howpublished={\url{https://github.com/countzerozzz/nodepert}} }
```
