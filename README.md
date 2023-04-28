# NodePert: A Perturbation-Based Method for Training Neural Networks 🧠

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python)](https://docs.python.org/3/whatsnew/3.11.html)
[![JAX](https://img.shields.io/badge/Framework-JAX-important?style=for-the-badge&logo=Apache-Kafka)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative)](https://github.com/countzerozzz/nodepert/edit/master/LICENSE.md)

[**Setup**](#setup)
| [**Running NodePert**](#running-nodepert)
| [**Paper Experiments**](figs/running-paper-exps.md)
<!-- | [**TF 1.x Repo**](https://github.com/yashsmehta/perturbations) -->

## Overview
Are you interested in learning more about the algorithms that drive goal-directed learning in the brain? Then you've come to the right place! This project explores how the brain may implement credit assignment by putting the ever-popular backpropagation and the ever-elusive node perturbation algorithms head-to-head.

In our repository, you'll find an efficient, scalable implementation of both node perturbation and backpropagation, on fully connected and convolutional networks. We've developed this code using JAX with Tensorflow Datasets (TFDS) to help with the data loading. So, why not take a deep dive into our repository and see what you can discover? We look forward to hearing your feedback!

## Setup

Can't wait to get your hands on our repository? Follow these simple steps, and you'll be up and running in no time (we hope):

1. Clone the repository like the git wizard you know you are:
```bash
git clone https://github.com/silverpaths/nodepert.git
```

2. Feast your eyes on the requirements.txt file for the project's dependencies. Install them with pip, but be warned—this installs JAX with CPU support only. Want to flex your GPU muscles? Uncomment the line in requirements.txt, if you dare.
```bash
pip install -r requirements.txt
```

3. To ensure JAX isn't just twiddling its virtual thumbs, change into the cloned directory, open a Python interpreter (as if you haven't done this a million times), and type:
```python
export PYTHONPATH <path/to/nodepert>
import fc_test
```

Run into any JAX installation snafus? Check out their official install guide for a helping hand. Best of luck, champ!

## Running NodePert

#### Going Solo: Running on a Single GPU/CPU Setup

Think you can handle NodePert? Let's find out. To train a basic network with node perturbation on MNIST, follow these oh-so-simple steps:

>For a Fully Connected Network (FC network architecture with 2 hidden layers and 500 neurons each):
>```python
>export PYTHONPATH=<path/to/nodepert>
>python fc_test.py -log_expdata True -n_hl 2 -hl_size 500 -lr 5e-3 -batchsize 100 -num_epochs 10 -update_rule np
>```
>For a Small Convolution Network (4 conv layers, 32 channels each, because why not?):
>```python
>export PYTHONPATH=<path/to/nodepert>
>python small_conv_test.py -log_expdata True -batchsize 100 -num_epochs 10 -update_rule sgd
>```
Don't forget to peek into the `utils.py` file for various arguments and default values. Use the `job_id` argument for parallel runs on a cluster, because who doesn't love multitasking?

#### Playing with the Big Boys: Running on a Compute Cluster with SLURM

Feeling adventurous? The slurm_scripts folder has you covered. Run experiments on a multinode setup with the SLURM resource manager. Use this handy-dandy command:
```bash
bash slurm-scripts/meta_jobscript.sh fc-test
```
#### Reliving the Glory: Running Experiments from the Paper
[check here](figs/running-paper-exps.md) for a trip down memory lane.

## Code Structure: The Nitty-Gritty
By default, we're all about MNIST, unless you change the data_loader in the `npimports.py` file (CIFAR10/100, f-MNIST). The experiments folder houses the code for all the experiments in the paper 'On the Limitations of Perturbation Based Methods for Training Deep Networks'. To run on your local machine (with a GPU or CPU, and a fully connected network), give this command a spin:

```python
python experiments/vary_lr.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 100 -update_rule np
```

And for all you neural network aficionados, take a gander at ```models/conv.py``` or ```models/fc.py```. The exact nodepert update can be found lurking in ```models/optim.py```. 

Happy coding!

