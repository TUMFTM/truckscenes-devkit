<div align="center">

<h1>MAN TruckScenes devkit</h1>

The first multimodal dataset for autonomous trucking

[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Windows](https://img.shields.io/badge/os-windows-blue.svg)](https://www.microsoft.com/windows/)

</div>

## Overview
- [Installation](#installation)
- [Setup](#truckscenes-setup)
- [Usage](#truckscenes-usage)
- [Citation](#citation)

## Installation
Our devkit is available and can be installed via pip:
```
pip install truckscenes-devkit
```

If you also want to install all the (optional) dependencies for running the visualizations:
```
pip install "truckscenes-devkit[all]"
```

The usage requires Python (install [here](https://www.python.org/downloads/), tested for 3.6 and 3.8) and pip (install [here](https://pip.pypa.io/en/stable/installation/)) for the installation.

## TruckScenes Setup
The MAN TruckScenes dataset can be downloaded on our [Download page](https://www.man.eu/truckscenes/) or search for TruckScenes in the [AWS Open Data Registry](https://registry.opendata.aws/).  

For the devkit to work you will need to download **all** archives.  
Please unpack the archives to the `/data/truckscenes` folder **without** overwriting folders that occur in multiple archives.  
Eventually you should have the following folder structure:
```
/data/truckscenes
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
```
If you want to use another folder, specify the `dataroot` parameter of the TruckScenes class (see [installation](./docs/installation.md)).

## TruckScenes Usage
Please follow these steps to make yourself familiar with the MAN TruckScenes dataset:
- [Download](https://www.man.eu/truckscenes/) the dataset on our website.
- Make yourself familiar with the [dataset schema](./docs/schema_truckscenes.md)
- Run the [tutorial](./python-sdk/tutorials/truckscenes_tutorial.ipynb) to get started:
  ```
  jupyter notebook $HOME/truckscenes-devkit/tutorials/truckscenes_tutorial.ipynb
  ```
- Read the [MAN TruckScenes Paper](https://arxiv.org/abs/2407.07462) for a detailed analysis of the dataset.

## Citation
```
@article{truckscenes2024,
  title={MAN TruckScenes: A multimodal dataset for autonomous trucking in diverse conditions},
  author={Fent, Felix and Kuttenreich, Fabian and Ruch, Florian and Rizwin, Farija and
    Juergens, Stefan and Lechermann, Lorenz and Nissler, Christian and Perl, Andrea and
    Voll, Ulrich and Yan, Min and Lienkamp, Markus},
  journal={arXiv preprint arXiv:2407.07462},
  year={2024}
}
```

_Copied and adapted from [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)_
