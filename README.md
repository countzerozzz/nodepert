# NodePert: A Perturbation-Based Method for Training Neural Networks 🧠

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python)](https://docs.python.org/3/whatsnew/3.11.html)
[![JAX](https://img.shields.io/badge/Framework-JAX-important?style=for-the-badge&logo=Apache-Kafka)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative)](https://github.com/countzerozzz/nodepert/edit/master/LICENSE.md)

[**Setup**](#setup)
| [**Running NodePert**](#running-node-perturbation)
| [**Paper Experiments**](figs/running-paper-exps.md)
<!-- | [**TF 1.x Repo**](https://github.com/yashsmehta/perturbations) -->

## Overview
Are you interested in learning more about the algorithms that drive goal-directed learning in the brain? Then you've come to the right place! This project explores how the brain may implement credit assignment by putting the ever-popular backpropagation and the ever-elusive node perturbation algorithms head-to-head. In our repository, you'll find an efficient, scalable implementation of both node perturbation and backpropagation, on fully connected and convolutional networks. We've developed this code using JAX with Tensorflow Datasets (TFDS) to help with the data loading. So, why not take a deep dive into our repository and see what you can discover? We look forward to hearing your feedback!

## Setup

Can't wait to get your hands on node perturbation? Follow these simple steps, and you'll be up and running in no time (we hope):

1. Create a new conda environment and clone the repository like the git wizard you know you are:
```bash
conda create -n nodepert python=3.11
conda activate nodepert
git clone https://github.com/silverpaths/nodepert.git
```

2. Install JAX and required libraries:

    a. **CPU only**
    ```bash
    pip install --upgrade "jax[cpu]"
    pip install -r requirements.txt
    ```
    b. **GPU**
    ```
    conda install -c nvidia cuda-toolkit
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -r requirements.txt
    ```

3. To ensure JAX isn't just twiddling its virtual thumbs and the installation is working properly, run a basic experiment on a fully connected network and a small convolution network:
```python
python fc_test.py
python small_conv_test.py
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
The experiments folder houses the code for all the experiments in the paper 'On the Limitations of Perturbation Based Methods for Training Deep Networks'. To run on your local machine (with a GPU or CPU, and a fully connected network), give this command a spin:

```python
python experiments/vary_lr.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 100 -update_rule np
```

And for all you neural network aficionados, take a gander at ```models/conv.py``` or ```models/fc.py```. The exact nodepert update can be found lurking in ```models/optim.py```.

#### Running on a Compute Cluster with SLURM
If you're looking to make the most out of your resources while running experiments on a multinode setup, then the `job_id` argument for parallel runs on a cluster can be a real lifesaver. It lets you multitask like a pro and get your experiments done faster. And with the SLURM resource manager, you can manage all your nodes seamlessly without breaking a sweat. So, why not give it a shot? 
```bash
bash slurm-scripts/meta_jobscript.sh fc-test
```
Best of luck, champ!

[**Running Experiments from the Paper**](figs/running-paper-exps.md)

## Citing
If you use this code in your own work, please use the following bibtex entry:

```bibtex
@misc{nodepert-2023,
  title={NodePert: On the Limitations of Perturbation Based Methods to Train Deep Neural Networks}, 
  author={Mehta, Yash and Hiratani, Naoki and Humphreys, Peter and Latham, Peter and Lillicrap, Tim}, 
  year={2023}, publisher={GitHub},
  howpublished={\url{https://github.com/countzerozzz/nodepert}} }
```
