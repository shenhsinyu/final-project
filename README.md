# final-project

## Global Wheat Detection
A competetiton on kaggle, the goal is to detect the wheat heads and draw bounding boxes around all the detected wheat heads in the given image.

## Environment
- ubuntu 18.04
- pytorch

## dataset
- join the competition and download the dataset.
- can use `data-augmentation.ipynb` to generate augmentation data.

## training
- run `train.py` to train your model for no augmentatino data.
- run `train_pseudo.py` to train on new data.

## parameters
- use can choose learning rate, epochs, batchsize as you want.

## result
- get 0.6583 on kaggle public score.
- image

 ![image](https://github.com/shenhsinyu/final-project/blob/main/image.png)

## reference
- https://github.com/Kaushal28/Global-Wheat-Detection
