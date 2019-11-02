import os
from setuptools import find_packages, setup


setup(
    name='AlgoCollection',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    description='',
    url='',
    author='',
    license='',

    install_requires=[
        'scikit-learn',
        'umap-learn',
    ]
)