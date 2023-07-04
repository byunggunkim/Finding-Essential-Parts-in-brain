import tensorflow as tf
from tensorflow import keras
import numpy as np


class Additive_attention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation1 = keras.activations.get('tanh')
        self.activation2 = keras.activations.get('sigmoid')

    def build(self, batch_input_shape):
        self.W_1 = self.add_weight(
            name="Weight_quary", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.W_2 = self.add_weight(
            name="Weight_key", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.W_3 = self.add_weight(
            name="Weight_score", shape=[self.units, 1],
            initializer="glorot_normal")
        self.bias1 = self.add_weight(
            name="bias_g", shape=[self.units], initializer="zeros")
        self.bias2 = self.add_weight(
            name="bias_score", shape=[1], initializer="zeros")
        super().build(batch_input_shape)

    def call(self, X):
        g_1 = tf.expand_dims(X @ self.W_1, axis=2) + tf.expand_dims(X @ self.W_2, axis=1) + self.bias1

        # (batch, 116, 116, hidden)
        g = self.activation1(g_1)

        # (batch, 116, 116, 1)
        score = self.activation2(g @ self.W_3 + self.bias2)

        # (batch, 116, 116) * (batch, 116, hidden)
        output = tf.squeeze(score, axis=-1) @ X

        return output

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape


class SCCNN_bn(keras.layers.Layer):
    def __init__(self, kernals, stride, padding, filters, activation, activation_param, init, l2, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv1D(filters[0], kernals[0], stride, padding, activation=None, input_shape=(None,1), kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.bn1 = keras.layers.BatchNormalization()

        if activation=='leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(activation_param)
        else:
            self.activation = tf.keras.layers.Activation(activation)

        # sccnn block
        self.pool1 = keras.layers.MaxPool1D(2,2)
        self.conv2 = keras.layers.Conv1D(filters[1], kernals[1], stride, padding, activation=None, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.bn2 = keras.layers.BatchNormalization()
        self.pool2 = keras.layers.MaxPool1D(2,2)
        self.conv3 = keras.layers.Conv1D(filters[2], kernals[2], stride, padding, activation=None, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.conv4 = keras.layers.Conv1D(filters[3], kernals[3], stride, padding, activation=None, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.bn3 = keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()
        self.pool3 = keras.layers.GlobalAveragePooling1D()

        self.out_filter = filters[3]

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.pool3(x)

        return x

    def compute_output_shape(self, batch_input_shape):
        return [None, self.out_filter]


class SCCNN_dilate(keras.layers.Layer):
    def __init__(self, kernals, stride, padding, filters, activation, activation_param, init, l2, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv1D(filters[0], kernals[0], stride, padding='valid', activation=None, input_shape=(None,1),kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.bn1 = keras.layers.BatchNormalization()
        self.pool1 = keras.layers.MaxPool1D(2,2)

        if activation=='leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(activation_param)
        else:
            self.activation = tf.keras.layers.Activation(activation)

        # Dilated cnn block
        self.conv2 = keras.layers.Conv1D(filters[1], kernals[1], stride, 'same', dilation_rate=2, activation=None, input_shape=(None,1),kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv1D(filters[2], kernals[2], stride, 'same', dilation_rate=2, activation=None, input_shape=(None,1),kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv1D(filters[3], kernals[3], stride, 'same', dilation_rate=2, activation=None, input_shape=(None,1),kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.bn4 = keras.layers.BatchNormalization()

        # Residual 
        self.conv_r = keras.layers.Conv1D(filters[3], kernals[3], stride, 'same', dilation_rate=2, activation=None, input_shape=(None,1),kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.bn_r = keras.layers.BatchNormalization()
        self.pool = keras.layers.GlobalAveragePooling1D()

        self.out_filter = filters[3]

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        traditional_cnn = self.pool1(x)

        # UFLB
        x = self.conv2(traditional_cnn)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)

        # Residual
        x_r = self.conv_r(traditional_cnn)
        x_r = self.bn_r(x_r)

        output = self.activation(x+x_r)
        output = self.pool(output)

        return output

    def compute_output_shape(self, batch_input_shape):
        return [None, self.out_filter]
