import numpy as np
import matplotlib.pyplot as plt


def plot_loss(his, ds):
    """
    :param his: The history of training returned by fit() method
    :param ds: the name of the dataset used
    """

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his['loss'])), his['loss'], label='train loss')
    plt.plot(np.arange(len(his['val_loss'])), his['val_loss'], label='valid loss')
    plt.title(ds + ' training loss')
    plt.legend(loc='best')
    plt.savefig('./plots/' + ds + '_his_loss.png')


def plot_acc(his, ds):
    """
    :param his: The history of training returned by fit() method
    :param ds: the name of the used dataset used
    """

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his['accuracy'])), his['accuracy'], label='train accuracy')
    plt.plot(np.arange(len(his['val_accuracy'])), his['val_accuracy'], label='valid accuracy')
    plt.title(ds + ' training accuracy')
    plt.legend(loc='best')
    plt.savefig('./plots/' + ds + '_his_acc.png')
