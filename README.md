<div align="center">

<h1>MAN TruckScenes devkit</h1>

The first multimodal dataset for autonomous trucking

[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Windows](https://img.shields.io/badge/os-windows-blue.svg)](https://www.microsoft.com/windows/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-blue.svg)](https://arxiv.org/abs/2407.07462)

[![Watch the video](https://private-user-images.githubusercontent.com/58326831/392028903-651a88be-f698-469b-9059-13374eddb9ae.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzMyNDAxMTQsIm5iZiI6MTczMzIzOTgxNCwicGF0aCI6Ii81ODMyNjgzMS8zOTIwMjg5MDMtNjUxYTg4YmUtZjY5OC00NjliLTkwNTktMTMzNzRlZGRiOWFlLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMDMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjAzVDE1MzAxNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJmNDJhODQ0OWM4MzZmMjk1MzI5NWFkYzBkNWE2ZWZiYmRiOTlkMWI2MTlhODM0ODg2NTJlOTc0Nzk5YzM3YWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.5OYKZZ7lkcxoP8FGQkiDHdtoTXwYG_0P4J5BwQ11tgE)](https://cdn-assets-eu.frontify.com/s3/frontify-enterprise-files-eu/eyJwYXRoIjoibWFuXC9maWxlXC9lb2s3TGF5V1RXMXYxZU1TUk02US5tcDQifQ:man:MuLfMZFfol1xfBIL7rNw0W4SqczZqwTuzhvI-yxJmdY?width={width}&format=mp4)

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
