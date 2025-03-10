<div align="center">

<h1>MAN TruckScenes devkit</h1>

World's First Public Dataset For Autonomous Trucking

[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Windows](https://img.shields.io/badge/os-windows-blue.svg)](https://www.microsoft.com/windows/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-blue.svg)](https://arxiv.org/abs/2407.07462)

[![Watch the video](https://raw.githubusercontent.com/ffent/truckscenes-media/main/thumbnail.jpg)](https://cdn-assets-eu.frontify.com/s3/frontify-enterprise-files-eu/eyJwYXRoIjoibWFuXC9maWxlXC9lb2s3TGF5V1RXMXYxZU1TUk02US5tcDQifQ:man:MuLfMZFfol1xfBIL7rNw0W4SqczZqwTuzhvI-yxJmdY?width={width}&format=mp4)

</div>

## Overview
- [Website](#🌐-website)
- [Installation](#💾-installation)
- [Setup](#🔨-setup)
- [Usage](#🚀-usage)
- [Citation](#📄-citation)

## 🌐 Website
To read more about the dataset or download it, please visit [https://www.man.eu/truckscenes](https://www.man.eu/truckscenes)

## 💾 Installation
Our devkit is available and can be installed via pip:
```
pip install truckscenes-devkit
```

If you also want to install all the (optional) dependencies for running the visualizations:
```
pip install "truckscenes-devkit[all]"
```

For more details on the installation see [installation](./docs/installation.md)

## 🔨 Setup
Download **all** archives from our [download page](https://www.man.eu/truckscenes/) or the [AWS Open Data Registry](https://registry.opendata.aws/).  

Unpack the archives to the `/data/man-truckscenes` folder **without** overwriting folders that occur in multiple archives.  
Eventually you should have the following folder structure:
```
/data/man-truckscenes
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
```

## 🚀 Usage
Please follow these steps to make yourself familiar with the MAN TruckScenes dataset:
- Read the [dataset description](https://www.man.eu/truckscenes/).
- Explore the dataset [videos](https://cdn-assets-eu.frontify.com/s3/frontify-enterprise-files-eu/eyJwYXRoIjoibWFuXC9maWxlXC9lb2s3TGF5V1RXMXYxZU1TUk02US5tcDQifQ:man:MuLfMZFfol1xfBIL7rNw0W4SqczZqwTuzhvI-yxJmdY?width={width}&format=mp4).
- [Download](https://www.man.eu/truckscenes/) the dataset from our website.
- Make yourself familiar with the [dataset schema](./docs/schema_truckscenes.md)
- Run the [tutorial](./tutorials/truckscenes_tutorial.ipynb) to get started:
- Read the [MAN TruckScenes paper](https://arxiv.org/abs/2407.07462) for a detailed analysis of the dataset.

## 📄 Citation
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
