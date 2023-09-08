import os
import argparse
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from data import get_data
from model import Vgg
from plot import plot_loss, plot_acc
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="fer2013")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--plot_history", type=bool, default=True)
args = parser.parse_args()
his = None

if args.dataset == "fer2013":
    # load the data
    x, y = get_data('datasets/M-FER2013_cropped/train')
    # convert the labels to the one-hot format
    y = to_categorical(y).reshape(y.shape[0], -1)
    # split the data into training set and validation set.
    # 'stratify' is used to ensure the data distributions of training and validation sets are identical.
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.08, random_state=0, stratify=y)

    print("Load M-fer2013 dataset with {} train images and {} validation iamges successfully!".format(y_train.shape[0],
                                                                                                      y_valid.shape[0]))

    # Data augmentation on the training set
    train_generator = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest',
        data_format='channels_last',
    ).flow(x_train, y_train, batch_size=args.batch_size)

    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=args.batch_size)

    model = Vgg()

    sgd = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'], )

    callback = [
        # Uncomment ModelCheckpoint() if you want to save the weights.
        # ModelCheckpoint(
        #     monitor='val_accuracy',
        #     verbose=True,
        #     save_weights_only=False,
        #     filepath="./models/mfer{epoch:02d}",
        #     period=10
        # ),

        # a learning rate scheduler which monitors the performance on the validation set.
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.75, patience=5, min_lr=0.0001)
    ]
    his = model.fit_generator(train_generator, steps_per_epoch=len(y_train) // args.batch_size, epochs=args.epochs,
                              validation_data=valid_generator, validation_steps=len(y_valid) // args.batch_size,
                              callbacks=callback)
elif args.dataset == "ck":
    # load the data
    x, y = get_data('datasets/M-CK+_cropped/train')
    # convert the labels to the one-hot format
    y = to_categorical(y).reshape(y.shape[0], -1)
    # split the data into training set and validation set.
    # A larger ration for the validation set is used, since the M-CK+ is very small.
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, random_state=15, stratify=y)

    print("Load M-CK+ dataset with {} train images and {} validation iamges successfully!".format(y_train.shape[0],
                                                                                                  y_valid.shape[0]))

    # Data augmentation on the training set
    train_generator = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        shear_range=0.15,
        fill_mode='nearest',
        data_format='channels_last',
    ).flow(x_train, y_train, batch_size=args.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=args.batch_size)

    model = Vgg()

    sgd = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'], )

    callback = [
        # Uncomment ModelCheckpoint() if you want to save the weights.
        # ModelCheckpoint(
        #                 monitor='val_accuracy',
        #                 verbose=True,
        #                 save_weights_only=False,
        #                 # period=10,
        #                 save_best_only=True,
        #                 filepath="./models/mck{epoch:02d}",
        #                 ),

        # a learning rate scheduler which monitors the performance on the validation set.
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.75, patience=10, min_lr=0.0001)
    ]
    his = model.fit_generator(train_generator, steps_per_epoch=len(y_train) // args.batch_size, epochs=args.epochs,
                              validation_data=valid_generator, validation_steps=len(y_valid) // args.batch_size,
                              callbacks=callback)

if args.plot_history:
    plot_loss(his.history, args.dataset)
    plot_acc(his.history, args.dataset)
