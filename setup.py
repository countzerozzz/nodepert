import pathlib
from setuptools import setup
from setuptools import find_namespace_packages


setup(
        name='nodepert',
        version='0.1',
        description='An empirical study of node perturbation for training deep networks',  
        url='https://github.com/countzerozzz/nodepert',
        author='Yash Mehta, Timothy Lillicrap',
        author_email='yashsmehta95@gmail.com',
        license='MIT',
        packages=[
            'nodepert',
            'nodepert.model',
            'nodepert.network_init',
            'data_loaders',
        ],
        install_requires=pathlib.Path('requirements.txt').read_text().splitlines(),
)