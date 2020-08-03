# nodepert

nodepert is a set of experiments written in Python + JAX to explore the
node perturbation algorithm for training deep and wide neural networks. A tensorflow implementation of node perturbation is available here: 

https://github.com/yashsmehta/perturbations

## Installation

Pull the nodepert package from git via:

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

To run the basic vanilla node perturbation on MNIST, run:
>Fully Connected Network
>```python
>python nptest.py
>```
>Convolution Network
>```python
>python npconvtest.py
>```


There are 2 ways to go about running the code from the experiments. 
>For a single compute node setup (GPU or CPU), run:
```python
python experiments/vary_lr.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 200 -update_rule np
```
Check out the _utils.py_ file for the different arguments which may be passed along with their default values. The argument job_id is for running the same code with different parameters in parallel. Forget about it in the single node case. The default job_id is 0, which runs the code with the parameter specified in the 0 <sup>th</sup> index of the list named _rows_ in the code. 

Note that the entire code runs on MNIST by default, unless the data_loader is explicitly changed in the _npimports.py_ file.

>For a multinode setup with SLURM resource manager, make use of the slurm scripts, somewhat as such:
```bash
bash slurm-scripts/meta_jobscript.sh trial
```

## Debug

If you aren't able to import specific modules (for example, ModuleNotFoundError: No module named 'npimports'), try adding the nodepert directory to the PYTHONPATH environment varible (export PYTHONPATH <path/to/nodepert>).

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements
A thanks to the creators of JAX who were responsive in helping us get set up when the installation for JAX failed.

## License
The source code for this project is licensed under the [MIT license](LICENSE.md).
