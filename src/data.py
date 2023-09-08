from tqdm import tqdm
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def get_data(path):
    """
    read the data given the path of a dataset
    :param path: the path of the dataset
    :return: Two arrays of x and y, which have the same sizes as the dataset.
             x is the images. y is the corresponding labels of x.
    """

    expressions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    x = []
    y = []
    for i in tqdm(range(len(expressions))):
        expression_folder = os.path.join(path, expressions[i])
        images = os.listdir(expression_folder)
        for j in range(len(images)):
            # Grayscale and resize images
            img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 96), color_mode="grayscale")
            img = img_to_array(img)
            x.append(img)
            y.append(i)
    # normalize grayscale values to (0,1)
    x = np.array(x).astype('float32') / 255.
    y = np.array(y).astype('int')
    return x, y
