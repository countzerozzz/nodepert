# NodePert: A Perturbation Based Method for Training Deep Neural Networks

What algorithms drive goal directed learning in networks in the brain? In machine learning, networks are almost exclusively trained using stochastic gradient descent, which delivers errors tailored to each neuron in a network. However, computing such tailored errors requires complicated machinery that is unlikely to exist in the brain, at least not in all areas. An alternative is to apply random perturbations to parameters, see whether that increases or decreases a global error, and then adjust parameters accordingly. The fact that this approach does not require complicated machinery, along with evidence that the brain utilizes global signals for learning, has prompted neuroscientists to speculate that the brain may use perturbation algorithms. However, little is known about the efficacy of these algorithms for training large networks to perform complex tasks. This repository contains the code for performing a thorough empirical investigation of a fast perturbation method, and see how it scales on large and convolutional network architectures and datasets for image classification tasks. These results provide insights into credit assignment in the brain.

The code was written using the [JAX Library](https://github.com/google/jax).

## Installation

Pull the repository from git via:

```bash
git clone https://github.com/silverpaths/nodepert.git
```

See requirements.txt file for the project dependencies, which can be installed via pip: 

```bash
pip3 install -r requirements.txt
```

Note that this will install tensorflow, which may attempt to install a different version of numpy & scipy than you have installed at present.

## Usage
To run a basic test, change into the cloned directory and open a Python
interpreter and issue:

```python
import nptest
```

#### Running on a single GPU/CPU setup
To train a basic network with node perturbation, run:
>Fully Connected Network (2 hidden layers, 500 neurons in each layer)
>```python
>python nptest.py
>```
>Small Convolution Network (4 conv layers, 32 channels in each layer)
>```python
>python npconvtest.py
>```

The entire code runs on MNIST by default, unless the data_loader is explicitly changed in the _npimports.py_ file. The experiments folder contains the code for all the experiments for the paper 'On the Limitations of Perturbation Based Methods for Training Deep Networks'. On your local machine (with GPU or CPU, and the default setup is a fully connected network), run:

```python
python experiments/vary_lr.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 100 -update_rule np
```

Check out the _utils.py_ file for the different arguments which may be passed along with their default values. The argument job_id is for running the same code with different parameters in parallel on a cluster.


#### Running on the SLURM cluster

The slurm_scripts folder contains the scripts to run experiments on a multinode setup with the SLURM resource manager. The default job_id is 0, which runs the code with the parameter specified in the 0 <sup>th</sup> index of the list named _rows_ in the code. Make use of the slurm scripts, run:

```bash
bash slurm-scripts/meta_jobscript.sh trial
```

## Debug
If you aren't able to import specific modules (for example, ModuleNotFoundError: No module named 'npimports'), try adding the nodepert directory to the PYTHONPATH environment varible (export PYTHONPATH <path/to/nodepert>).

## License
The source code for this project is licensed under the [MIT license](LICENSE.md).
