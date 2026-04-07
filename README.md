# fashion-mist-xai
CNN explainability on Fashion-MNIST using Grad-CAM and Integrated Gradients

This project investigates explainability for CNN-based image classification on the Fashion-MNIST dataset. It compares two explanation methods, Grad-CAM and Integrated Gradients, to examine how they highlight meaningful image regions for correct and incorrect predictions. The project was developed for an Explainable AI assignment.

## Dataset
Fashion-MNIST (TensorFlow Datasets)

## Methods
- Grad-CAM
- Integrated Gradients

## Requirements
pip install -r requirements.txt

## Run
python main.py

## Output
The script:
- trains a CNN on Fashion-MNIST
- evaluates test performance
- generates one correct and one incorrect explanation example
- saves:
  - `correct_example.png`
  - `incorrect_example.png`
