import tensorflow as tf


class Vgg(tf.keras.Model):
    def __init__(self, drop=0.30):
        super(Vgg, self).__init__()

        # Define the convolutional and batch normalization layers
        self.conv_bn_layers = []
        for i, filters in enumerate([32, 64, 128, 256]):
            self.conv_bn_layers.append(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu',
                                       name=f'conv{i + 1}a'))
            self.conv_bn_layers.append(tf.keras.layers.BatchNormalization())
            self.conv_bn_layers.append(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu',
                                       name=f'conv{i + 1}b'))
            self.conv_bn_layers.append(tf.keras.layers.BatchNormalization())
            self.conv_bn_layers.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

        self.flatten = tf.keras.layers.Flatten()
        self.lin1 = tf.keras.layers.Dense(units=4096, activation='relu')
        self.lin2 = tf.keras.layers.Dense(units=4096, activation='relu')
        self.drop = tf.keras.layers.Dropout(rate=drop)
        self.lin3 = tf.keras.layers.Dense(7, activation='softmax')

    def call(self, x):
        # Pass the input through the convolutional and batch normalization layers
        for layer in self.conv_bn_layers:
            x = layer(x)

        x = self.flatten(x)
        x = self.lin1(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.drop(x)
        x = self.lin3(x)
        return x
