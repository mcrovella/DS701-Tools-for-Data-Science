# DS 701: Tools for Data Science

Repository for DS 701 that produces the [course notes](https://mcrovella.github.io/DS701-Tools-for-Data-Science).

## Setup and Build Instructions

The setup instructions are particular to supporting the RISE package for displaying Jupyter notebooks as slides.

> These instructions are tailored to, and tested on, MacoS Ventura 13.5. If you
> want to run this on Windows, contact the course instructors.

### Conda Python 3.9 Installation

Installing the RISE package is known to work with a Python 3.9 environment and the associated Jupyter packages so for now we will use Python 3.9.

#### Install Conda
Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), if you haven't already.

#### Create a Conda Virtual Environemnt from Environment File

```sh
conda env create -f environment-dev.yml
```
Now activate the environment.

```sh
conda activate ds701_dev_env
```

After you successfully activated the environment, run this command from
the Conda virtual environment.

```sh
jupyter contrib nbextension install --user
```


#### Create a Conda Virtual Environment Manually

If you prefer to create the Conda environment manually, or the commands
above didn't work you can follow the instructions below.

Create a conda environment with Python 3.9:

```sh
# replace my_env with any name for your environment
conda create -n my_env python=3.9
conda activate my_env
```

##### Package Installation

Once your new environment is activated, install the following packages:

```sh
conda install notebook
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```
Then you can install the rest of the dependencies, one at a time.

```sh
conda install -c conda-forge widgetsnbextension
conda install -c conda-forge ipywidgets
pip install -U RISE
conda install -c conda-forge jupyter-book
conda install numpy
conda install matplotlib
conda install scipy
conda install pandas
conda install -c conda-forge seaborn
conda install -c conda-forge scikit-learn
conda install networkx
conda install -c conda-forge qrcode
pip install graphviz
```
## Starting Jupyter Notebooks

To start and run the Jupyter notebooks in this repo, type the following at a command prompt in your Conda environment.

```sh
jupyter notebook
```
That should open up the Jupyter notebook listing page in your default web browser.

From there you can click on any notebook to open it.

## Updating the Jupyter Book

This should normally only be done by the course instructors.

To create or update a local copy of the Jupyter Book, run the following command
from your Conda environment.

```sh
make book
```
