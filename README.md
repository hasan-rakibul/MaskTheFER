This repository contains all codes of our paper **MaskTheFER: Mask-Aware Facial Expression Recognition using Convolutional Neural Network**. Codes are primarily developed by Cheng Jiang.

# Raw and Masked Datasets
We conducted experiments on two FER databases: FER2013 and CK+. Before training and testing, a series of preprocessing operations were performed on these two databases.

## CK+
It is small and contains highly similar images as these images are consecutive frames from videos. Source: https://www.kaggle.com/datasets/shawon10/ckplus and https://github.com/spenceryee/CS229/tree/master/CK%2B

First we apply `remove_duplicate.py` on raw CK+ data. It is resized to approximately one-third of its original size because only one out of three frames from the same video is sampled.

## FER2013
Another facial expression recognition similar to CK+ with a larger size. Source: https://www.kaggle.com/datasets/msambare/fer2013

## MaskedFER2023 and MaskedCK+
**We have published the processed datasets at https://doi.org/10.17632/sp3xssmzbg.1.** Download and place the datasets according to the following folder structure:
```bash
datasets
├── M-CK+
├── M-CK+_cropped
├── M-FER2013
├── M-FER2013_cropped
```
`./datasets/M-FER2013` and `./datasets/M-CK+` are the masked version of FER2013 and CK+ datases. They are obtained by running MaskTheFace algorithm on FER2013 and CK+. The implementation of MaskTheFace can be found at https://github.com/aqeelanwar/MaskTheFace. You will need to install MaskTheFace and follow the instructions in its readme file. Additionally, it is important to overwrite its `utils/aux_functions.py` file with the one provided under the `/src/` folder in this repository. We have made some modifications to the parameters to ensure that algorithm can work on low-resolution grayscale images.

`./datasets/M-FER2013_cropped` and `./datasets/M-CK+_cropped` are the cropped version of `./datasets/M-FER2013` and `./datasets/M-CK+`, respectively by runningy `crop.py`. Only regions near the eyes are preserved, and some poorly masked images (inappropriate masked sizes and positions) are removed. These two datasets are then split into training and test sets, serving as inputs to the model.

Note that **MaskedFER2023** and **MaskedCK+** in the paper refer to **M-FER2013_cropped** and **M-CK+_cropped**, respectively, in this repository.

# How the model is trained?

## Arguments
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

# How to evaluate on test datasets?
Once the weights of the models are placed into the `models` folder, you can execute the `evaluate.ipynb` file to see the results.

# Citation
If you find this repository useful in your research, please cite our paper and the dataset as follows:
```
@inproceedings{jiang2023maskthefer,
    title={{MaskTheFER}: Mask-Aware Facial Expression Recognition using Convolutional Neural Network},
    author={Jiang, Cheng and Hasan, Md Rakibul and Gedeon, Tom and Hossain, Md Zakir},
    booktitle={2023 International Conference on Digital Image Computing: Techniques and Applications (DICTA)},
    year={2023},
    pages={1-8},
    organization={IEEE}
}
```
```
@misc{jiang2023maskedfer2023,
    title={{MaskedFER2023} and {MaskedCK}+: Face-masked {FER2013} and {CK+} datasets for mask-aware facial expression recognition},
    author={Jiang, Cheng and Hasan, Md Rakibul and Gedeon, Tom and Hossain, Md Zakir},
    howpublished={Mendeley Data},
    note={V1},
    year={2023},
    doi={10.17632/sp3xssmzbg.1}
}
```
