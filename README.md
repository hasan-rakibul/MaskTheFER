This repository contains all codes of our paper **MaskTheFER: Mask-Aware Facial Expression Recognition using Convolutional Neural Network**. Codes are primarily developed by Cheng Jiang.

## Datasets
We conducted experiments on two FER databases: FER2013 and CK+. Before training and testing, a series of preprocessing operations were performed on these two databases.

### CK+
It is small and contains many highly similar images as these images are consecutive frames from videos. Source: https://www.kaggle.com/datasets/shawon10/ckplus and https://github.com/spenceryee/CS229/tree/master/CK%2B

First we apply `remove_duplicate.py` on raw CK+ data. It is resized to approximately one-third of its original size because only one out of three frames from the same video is sampled.

### FER2013
Another facial expression recognition similar to CK+ with a larger size. Source: https://www.kaggle.com/datasets/msambare/fer2013

### MaskedFER2023 and MaskedCK+
`./datasets/M-FER2013` and `./datasets/M-CK+` are the masked version of FER2013 and CK+ datases. They are obtained by running MaskTheFace algorithm on FER2013 and CK+. The implementation of MaskTheFace can be found at https://github.com/aqeelanwar/MaskTheFace. You will need to install MaskTheFace and follow the instructions in its readme file. Additionally, it is important to overwrite its `utils/aux_functions.py` file with the one provided under the `/src/` folder in this repository. We have made some modifications to the parameters to ensure that algorithm can work on low-resolution grayscale images.

`./datasets/M-FER2013_cropped` and `./datasets/M-CK+_cropped` are the cropped version of `./datasets/M-FER2013` and `./datasets/M-CK+`, respectively by runningy `crop.py`. Only regions near the eyes are preserved, and some poorly masked images (inappropriate masked sizes and positions) are removed. These two datasets are then split into training and test sets, serving as inputs to the model.

Note that **MaskedFER2023** and **MaskedCK+** in the paper refers to **M-FER2013_cropped** and **M-CK+_cropped**, respectively, in this repository.

## How the model is trained?

### Arguments
|   Argument   |                                                                  Explanation                                                                  |
|:------------:|:---------------------------------------------------------------------------------------------------------------------------------------------:|
|   dataset    |                                                        Can be either 'fer2013' or 'ck'                                                        |
|    epochs    |                                          An integer,the number of epochs you want to train the model                                          |
|  batch_size  |                                An integer,the number of training examples processed together in one iteration                                 |
| plot_history |                                A boolean value, whether you wish to plot the loss and accuracy after training                                 |


```bash
# Example 1 with MaskedFER2023 dataset
python src/train.py --dataset fer2013 --epochs 150 --batch_size 64

# Example 2 with MaskedCK+ dataset
python src/train.py --dataset ck --epochs 150 --batch_size 16
```    

## How the model is evaluated?
Once the weights of the models are placed into the `models` folder, you can execute the `evaluate.ipynb` file to see the results.