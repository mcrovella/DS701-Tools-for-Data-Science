# DS 701: Tools for Data Science

Repository for DS 701 as taught by Mark Crovella - BU CDS

## Setup and Build Instructions

> These instructions are tailored to, and tested on, MacoS Ventura 13.5.

### Conda Python=3.9 Installation

#### Install Conda
Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), if you haven't already.

Create a conda environment with Python 3.9:

```sh
# replace my_env with any name for your environment
conda create -n my_env python=3.9
conda activate my_env
```
Once your new environment is activated, install the following:

```sh
conda install notebook
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
pip install -U RISE
```

### venv, pip installation

#### Pre-requisites

* python3 (tested on MacOS with Python 3.9.6)
* docker (see https://docker.com)

#### Setup Python Virtual Environment

Clone this repo, or if you think you might be suggesting changes, fork the repo
on github.com, and then clone from your forked repo.

Create and checkout a development branch to more cleanly track any modificaitons
you make.

```sh
git checkout -b my_branch_name
```

Create a new, empty virtual environment in your local folder. For example
```sh
python3 -m venv .venv
```

If you haven't already, install `pip-tools`.
```sh
pip install pip-tools
```

Now you should be able to generate the `requirements.txt` file by running
```sh
make requirements.txt
pip install -r requirements.txt
```

#### Create Docker Container

