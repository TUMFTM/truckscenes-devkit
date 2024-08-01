# Advanced Installation
We provide step-by-step instructions to install our devkit.
- [Download](#download)
- [Install Python](#install-python)
- [Setup a virtual environment](#setup-a-virtual-environment)
- [Setup PYTHONPATH](#setup-pythonpath)
- [Install required packages](#install-required-packages)
- [Setup environment variable](#setup-environment-variable)
- [Setup Matplotlib backend](#setup-matplotlib-backend)
- [Verify install](#verify-install)

## Download

Download the devkit to your home directory.

## Install Python

The devkit is tested for Python 3.6 onwards, but we recommend to use Python 3.8.
For Ubuntu: If the right Python version is not already installed on your system, install it by running:
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8
sudo apt install python3.8-dev
```
For Windows or Mac OS see `https://www.python.org/downloads/`.

-----
## Setup a virtual environment
For setting up a virtual environment we use venv.

#### Install venv
To install venv, run:
```
sudo apt-get install python3-venv 
```

#### Create the virtual environment
We create a new virtual environment named `venv`.
```
python3 -m venv venv
```

#### Activate the virtual environment
If you are inside the virtual environment, your shell prompt should look like: `(venv) user@computer:~$`
If that is not the case, you can enable the virtual environment using:
```
source venv/bin/activate
```
To deactivate the virtual environment, use:
```
deactivate
```

## Setup PYTHONPATH
Add the `src` directory to your `PYTHONPATH` environmental variable:
```
export PYTHONPATH="${PYTHONPATH}:$HOME/truckscenes-devkit/src"
```

## Install required packages

To install the required packages, run the following command in your favorite virtual environment:
```
pip install -r requirements.txt
```
**Note:** The requirements file is internally divided into base requirements (`base`) and additional requirements (`tutorial`, `visu`).

If you want to install these additional requirements, please run:
```
pip install -r setup/requirements/requirements_<>.txt
``` 

## Setup environment variable
Finally, if you want to run the unit tests you need to point the devkit to the `truckscenes` folder on your disk.
Set the TRUCKSCENES environment variable to point to your data folder:
```
export TRUCKSCENES="/data/truckscenes"
```

## Setup Matplotlib backend
When using Matplotlib, it is generally recommended to define the backend used for rendering:
1) Under Ubuntu the default backend `Agg` results in any plot not being rendered by default. This does not apply inside Jupyter notebooks.
2) Under MacOSX a call to `plt.plot()` may fail with the following error (see [here](https://github.com/matplotlib/matplotlib/issues/13414) for more details):
    ```
    libc++abi.dylib: terminating with uncaught exception of type NSException
    ```
To set the backend, add the following to your `~/.matplotlib/matplotlibrc` file, which needs to be created if it does not exist yet: 
```
backend: TKAgg
```

## Verify install
To verify your environment run `python -m unittest` in the `test` folder.

That's it you should be good to go!





Copied and adapted from [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)