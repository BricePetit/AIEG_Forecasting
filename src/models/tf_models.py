"""
In this module, we define the models that will be used in the project.
We have the seq2point (https://dl.acm.org/doi/abs/10.5555/3504035.3504353) model and
the UNet model (https://dl.acm.org/doi/10.1145/3427771.3427859).

This module defines the models using TensorFlow.
"""
__title__: str = "models_tensorflow"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# Imports standard libraries

# Imports third party libraries
import tensorflow as tf

# Imports from src

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class TFSeq2Point(tf.keras.Model):
    """
    TensorFlow's implementation of the Seq2Point model.
    """
    def __init__(self, sequence_length: int, *args, **kwargs):
        """
        Constructor / Initializer of the TFSeq2Point class.

        Original code: https://github.com/MingjunZhong/seq2point-nilm

        :param sequence_length: Sequence length.
        """
        super().__init__(*args, **kwargs)
        self.model = tf.keras.models.Sequential([
            # Input layer is preferred instead of input_shape param.
            tf.keras.Input(shape=(sequence_length, 1)),
            # First value is the number of filters, second value is the kernel size (filter size)
            # The first value represents the number of element in the output
            tf.keras.layers.Conv1D(
                30, 10, activation='relu', strides=1
            ),
            tf.keras.layers.Conv1D(30, 8, activation='relu', strides=1),
            tf.keras.layers.Conv1D(40, 6, activation='relu', strides=1),
            tf.keras.layers.Conv1D(50, 5, activation='relu', strides=1),
            tf.keras.layers.Conv1D(50, 5, activation='relu', strides=1),
            # tf.keras.layers.Conv1D(
            #     30, 8, activation='relu', input_shape=(self.sequence_length, 1), strides=1
            # ),
            # tf.keras.layers.Conv1D(30, 6, activation='relu', strides=1),
            # tf.keras.layers.Conv1D(40, 4, activation='relu', strides=1),
            # tf.keras.layers.Conv1D(50, 3, activation='relu', strides=1),
            # tf.keras.layers.Conv1D(50, 3, activation='relu', strides=1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

    def call(self, inputs, training=None, mask=None):
        """
        Function to apply the forward pass.

        :param inputs:      Inputs tensor or /dic/list/tuple of tensor.
        :param training:    Training mode.
        :param mask:        Mask.

        :return:        Return the output of the model.
        """
        return self.model(inputs)


class TFUNetNilm(tf.keras.Model):
    """
    TensorFlow's implementation of the UNet model.
    """
    def __init__(self, sequence_length: int, *args, **kwargs):
        """
        Constructor / Initializer of the TFUNetNilm class.

        :param sequence_length: Length of the sequence.
        """
        super().__init__(*args, **kwargs)
        pool_filter = 8
        output_size = 1
        self.enc_layers = [
            tf.keras.Sequential([
                tf.keras.Input(shape=(sequence_length, 1)),
                tf.keras.layers.Conv1D(30, 10, 2, activation='relu')
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(30, 8, 2, activation='relu')
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(40, 6, 1, activation='relu')
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(50, 5, 1, activation='relu'),
                tf.keras.layers.Dropout(0.2)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(50, 5, 1, activation='relu')
            ])
        ]

        self.dec_layers = [
            TFUpLayer(40, 5, 1),
            TFUpLayer(30, 5, 1),
            TFUpLayer(30, 6, 1)
        ]

        self.fc = tf.keras.Sequential([
            tf.keras.layers.AveragePooling1D(pool_filter),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output_size, activation='linear')
        ])

    def call(self, inputs, training=None, mask=None):
        """
        Function to apply the forward pass.

        :param inputs:      Inputs tensor or /dic/list/tuple of tensor.
        :param training:    Training mode.
        :param mask:        Mask.
        """
        xi = [self.enc_layers[0](inputs)]

        for layer in self.enc_layers[1:]:
            xi.append(layer(xi[-1]))

        for i, layer in enumerate(self.dec_layers):
            xi[-1] = layer((xi[-1], xi[-2 - i]))
        out = self.fc(xi[-1])
        return out


class TFUpLayer(tf.keras.Model):
    """
    Class for the Up layer in Tensorflow.
    """

    def __init__(self, filters, kernel, stride, *args, **kwargs):
        """
        Constructor / Initializer of the Up class.

        :param filters:     Filters.
        :param kernel:      Kernel size.
        :param stride:      Stride.
        """
        super().__init__(*args, **kwargs)
        self.upsample = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(filters, kernel, stride, activation='relu')
        ])
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters, kernel, stride, activation='relu')
        ])

    def call(self, inputs, training=None, mask=None):
        """
        Function to apply the forward pass.

        :param inputs:      Tensor containing the input (x1 and x2).
        :param training:    Training mode.
        :param mask:        Mask.

        :return:    Return a tensor containing the output.
        """
        x1, x2 = inputs[0], inputs[1]
        x1 = self.upsample(x1)
        # Get the dynamic shapes of x1 and x2
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # Calculate the padding size
        diff = x2_shape[1] - x1_shape[1]
        pad_before = diff // 2
        pad_after = diff - pad_before
        # Pad x1 to the size of x2
        x1 = tf.pad(x1, [[0, 0], [pad_before, pad_after], [0, 0]])
        # Concatenate along the channels axis
        x = tf.concat([x2, x1], axis=-1)
        x = self.conv(x)
        return x
