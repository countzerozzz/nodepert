# 🧠 NodePert: Perturbation-Based Algorithms for Training Deep Neural Networks

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python)](https://docs.python.org/3/whatsnew/3.11.html)
[![JAX](https://img.shields.io/badge/Framework-JAX-important?style=for-the-badge&logo=Apache-Kafka)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative)](https://github.com/countzerozzz/nodepert/edit/master/LICENSE.md)

[**Setup**](#setup)
| [**Running NodePert**](#running-node-perturbation)
| [**Paper**]()

## Overview
What algorithms underlie goal directed learning in the brain? Backpropagation is the standard **credit assignment algorithm** used in machine learning research, but it's considered biologically implausible. Recently, **biologically plausible** alternatives, such as feedback alignment, target propagation, and perturbation algorithms, have been explored. The node perturbation algorithm applies random perturbations to neuron activity, monitors performance, and adjusts weights accordingly. This approach is simple and may be utilized by the brain. 

This repository contains the accompanying code for the paper, *An empirical study of perturbation methods for training deep networks*. It offers a *efficient* and *scalable* implementation of perturbation algorithms, allowing for large-scale experiments with node perturbation on **modern convolutional architectures on a GPU**. Our results provide insights into the diverse credit assignment algorithms used by the brain. The code was written by [Yash Mehta](https://yashsmehta.github.io/) and [Timothy Lillicrap](https://contrastiveconvergence.net/~timothylillicrap/index.php) using [**`JAX`**](https://github.com/google/jax) in conjunction with `Tensorflow Datasets` for data loading. Reach out to yashsmehta95[at]gmail.com or timothy.lillicrap[at]gmail.com with queries or feedback.

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
    Based on your CUDA version, check if you need to use "jax[cuda11_pip]" or "jax[cuda12_pip]"
    
    ```bash
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -e .
    ```
    
    </details>
    

3. To ensure JAX is working properly, run a basic experiment on a fully connected network comparing node perturbation and SGD on MNIST data. This saves a learning plot, and should less than 2m to run.

    ```python
    python example.py
    ```

Run into any JAX installation snafus? Check out their [**official install guide**](https://github.com/google/jax#installation) for a helping hand.

## Running Node Perturbation

You can customize the entire training process by passing different arguments to a single file, `main.py`. An example of argparse parameters include:

- **dataset**: `mnist`, `fmnist`, `cifar10` 
- **network**: `fc`, `linfc`, `conv`, `conv-large`
- **update rule**: `np`, `sgd`

For a full list of parameters and default values, refer to the `parse_args()` function in `utils.py`. To see an example of how to run the training process with your desired arguments, you can use the `main.py` file.
>```python
>python nodepert/main.py -network fc -dataset mnist -log_expdata True -n_hl 2 -hl_size 500 -lr 5e-3 -batchsize 100 -num_epochs 10 -update_rule np
>```
>

#### Detailed experiments

Inside the experiments folder, you'll find example code for a variety of experiments utilizing node perturbation, for example:

1. **Understanding network crashes during training**. See `crash-dynamics.py`, `crash_timing.py`, `grad_dynamics.py`
2. **Relative change in the loss with different learning rates**. See `linesearch.py`, `linesearch_utils.py`
3. **Adam-like update for NP gradients**. See `adam_update.py`
4. **Visualizing the loss landscape**. See `loss_landscape.py`

And for all you neural network aficionados, take a gander at ```model/conv.py``` or ```model/fc.py```. The exact nodepert update can be found in ```optim.py```.

#### Running on a compute cluster
You can directly run multiple configurations with ease by simply specifying values in a dictionary in `cluster_scripts/scheduler.py`. It schedules *all* combinations of hyperparameters specified in the dictionary, along with multiple seeds of your experiments simultaneously. This is extremely useful for GPU clusters that have resource allocation managers like SLURM.

```bash
bash slurm-scripts/scheduler.py
```

## Citing
If you use this code in your own work, please use the following bibtex entry:

```bibtex
@misc{nodepert-2023,
  title={NodePert: An empircal study of perturbation methods for training deep networks}, 
  author={Mehta, Yash and Hiratani, Naoki and Humphreys, Peter and Latham, Peter and Lillicrap, Timothy}, 
  year={2023}, publisher={GitHub},
  howpublished={\url{https://github.com/countzerozzz/nodepert}} }
```
