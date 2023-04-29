from setuptools import setup

setup(
        name='nodepert',
        version='0.1',
        description='Large-scale experiments with node perturbations',
        url='https://github.com/countzerozzz/nodepert',
        author='Yash Mehta, Timothy Lillicrap',
        author_email='yashsmehta95@gmail.com',
        license='MIT',
        packages=[
            'models',
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