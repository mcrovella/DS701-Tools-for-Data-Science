# DS 701: Tools for Data Science

Repository for DS 701 that produces the [course notes](https://mcrovella.github.io/DS701-Tools-for-Data-Science).

## Setup and Build Instructions

The setup instructions are particular to supporting the RISE package for displaying Jupyter notebooks as slides.

> These instructions are tailored to, and tested on, MacoS Ventura 13.5.

### Conda Python 3.9 Installation

Installing the RISE package is known to work with a Python 3.9 environment and the associated Jupyter packages so for now we will use Python 3.9.

#### Install Conda
Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), if you haven't already.

Create a conda environment with Python 3.9:

```sh
# replace my_env with any name for your environment
conda create -n my_env python=3.9
conda activate my_env
```

#### Package Installation

Once your new environment is activated, install the following packages:

```sh
conda install notebook
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
pip install -U RISE
conda install -c conda-forge jupyter-book
conda install numpy
```
