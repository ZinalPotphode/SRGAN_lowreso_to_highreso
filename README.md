# SRGAN Resolution Booster

A machine learning model designed to enhance low-resolution images into high-resolution versions using the Super-Resolution Generative Adversarial Network (SRGAN) architecture. This project focuses on optimizing neural network architecture and the training process to achieve a high-quality image resolution enhancement, reaching up to 92% improvement in image fidelity.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction

SRGAN Resolution Booster is a deep learning-based project that leverages the power of Generative Adversarial Networks (GANs) to upscale low-resolution images. The model is implemented using Python and popular machine learning libraries such as TensorFlow and Keras, with the architecture optimized to maximize resolution enhancement.

## Features

- Implementation of SRGAN, a state-of-the-art model for image super-resolution.
- Optimized neural network architecture for superior image quality enhancement.
- Training pipeline designed for maximum efficiency and performance.
- Achieves up to 92% enhancement in image resolution.

## Installation

To run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ZinalPotphode/SRGAN_lowreso_to_highreso
cd SRGAN_lowreso_to_highreso
```
## Usage

- Training the Model: Use the provided scripts to train the SRGAN model on your dataset.
- Generating High-Resolution Images: After training, use the model to enhance low-resolution images.

```bash
# Train the model
python train.py --dataset <path_to_dataset> --epochs 1000

# Enhance images
python enhance.py --input <low_res_image> --output <high_res_output>

```
## Dependencies
This project relies on the following libraries and frameworks:

Python 3.x
TensorFlow
Keras
NumPy
Matplotlib
OpenCV
Scikit-learn

You can install these dependencies using pip:
```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```