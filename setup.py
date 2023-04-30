import pathlib
from setuptools import setup
from setuptools import find_namespace_packages


setup(
        name='nodepert',
        version='0.1',
        description='An empirical study of the node perturbation',
        url='https://github.com/countzerozzz/nodepert',
        author='Yash Mehta, Timothy Lillicrap',
        author_email='yashsmehta95@gmail.com',
        license='MIT',
        packages=[
            'model',
            'data_loaders',
        ],
        install_requires=[
            'tensorflow-cpu',
            'tensorflow-datasets',
            'numpy',
            'pandas',
            'scikit-learn',
            'matplotlib',
            'protobuf==3.20.*'
        ]
)