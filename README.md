# APACS23
This repository contains the codebase required to train neural network models on our APACS23 dataset.

The following networks are implemented:
- FCN-32
- FCN-16
- FCN-8
- methods proposed in our paper (Bogacsovics, G., Hajdu, A., & Harangi, B. (2021, December). Cell Segmentation in Digitized Pap Smear Images Using an Ensemble of Fully Convolutional Networks. In 2021 IEEE Signal Processing in Medicine and Biology Symposium (SPMB) (pp. 1-6). IEEE.).

The repository also contains the various helper scripts (e.g. for loading and normalizing images) in the files named helper.py and utils.py.

# Usage notes
To run the script, use the command
`python main.py`

The user can use the config.yaml file to change between training and test modes, set hyperparameters, number of classes (1=binary) and configure the input and output directories.

## Training

Use `mode: "training"`, configure the batch size and set the dataset locations (both inputs and ground truths) under the `datasets` section. Finally, set the number of epochs and learning rate under the `modes: training` section.

## Test
Use `mode: "test"`, configure the batch size and set the dataset locations (both inputs and ground truths) under the `datasets` section. Finally, set the checkpoint location (".pth" file), saving directory, and an optinal tag under the `modes: test` section.
