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

## Usage

To run a basic test, change into the cloned direction and open a Python
interpreter. Then issue:

```python
import nptest
```
If you aren't able to import specific modules (for example, ModuleNotFoundError: No module named 'npimports'), try adding the nodepert directory to the PYTHONPATH environment varible (export PYTHONPATH <path/to/nodepert>).

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements
A thanks to the creators of JAX who were responsive in helping us get set up when the installation for JAX failed.

## License
???
