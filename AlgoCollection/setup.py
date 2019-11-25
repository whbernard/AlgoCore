import os
from setuptools import find_packages, setup


setup(
    name='AlgoCollection',
    version='0.1.0',
    packages=find_packages(where='src', exclude=["test"]),
    package_dir={'': 'src'},
    include_package_data=True,
    description='',
    url='',
    author='',
    license='',

    install_requires=[
        'scikit-learn',
        'umap-learn',
        'pandas',
    ]
)