import tensorflow as tf
from tensorflow import keras
import numpy as np
import layers


class SC_CNN_Attention_roi(keras.Model):
    def __init__(self, kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attend_hidden, dense_hidden, init, l2_init, drop_rate, roi, **kwargs):
        super().__init__(**kwargs)

        self.roi_id = roi
        self.scrnn = layers.SCCNN_bn(kernals=kernal_list, stride=1, padding=padding, filters=filter_list, activation=cnn_activation, activation_param=activation_param, init=init, l2=l2_init, rate=drop_rate)

        self.attention = layers.Additive_attention(units=attend_hidden)
        self.dense = tf.keras.layers.Dense(dense_hidden, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(l2_init))
        self.out = keras.layers.Dense(2,activation='softmax', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(l2_init))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=False):
        inputs = inputs.to_tensor()
        inputs = tf.expand_dims(inputs, axis=-1)

        Encoder = [self.scrnn(inputs[:,idx]) for idx in self.roi_id]
        concated =tf.stack(Encoder, axis=1)
        attention_out = self.attention(concated)
        attended_flat = keras.layers.Flatten()(attention_out)

        dense_out = self.dense(attended_flat)
        dense_out = self.dropout(dense_out, training=training)
        output = self.out(dense_out)

        return output


class SC_CNN_LSTM_roi(keras.Model):
    def __init__(self,kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2_init, drop_rate, roi, **kwargs):
        super().__init__(**kwargs)

        self.roi_id = roi
        self.scrnn = layers.SCCNN_bn(kernals=kernal_list, stride=1, padding=padding, filters=filter_list, activation=cnn_activation, activation_param=activation_param, init=init, l2=l2_init, rate=drop_rate)

        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_hidden, return_sequences=False, time_major=False))

        self.dense = tf.keras.layers.Dense(dense_hidden, activation='relu',kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(l2_init))
        # self.bn = keras.layers.BatchNormalization()
        self.out = keras.layers.Dense(2,activation='softmax', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(l2_init))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=False):
        inputs = inputs.to_tensor()
        inputs = tf.expand_dims(inputs, axis=-1)
        
        Encoder = [self.scrnn(inputs[:,idx]) for idx in self.roi_id]
        concated =tf.stack(Encoder, axis=1)
        seq_out = self.lstm(concated)

        dense1 = self.dense(seq_out)
        dense1 = self.dropout(dense1, training=training)
        output = self.out(dense1)

        return output


class SC_CNN_acrnn_roi(keras.Model):
    def __init__(self,kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2_init, drop_rate, roi, **kwargs):
        super().__init__(**kwargs)

        self.roi_id = roi
        self.scrnn = layers.SCCNN_bn(kernals=kernal_list, stride=1, padding=padding, filters=filter_list, activation=cnn_activation, activation_param=activation_param, init=init, l2=l2_init, rate=drop_rate)

        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_hidden, return_sequences=True, time_major=False))

        self.attention = layers.Additive_attention(units=attention_hidden)
        self.dense = tf.keras.layers.Dense(dense_hidden, activation='relu',kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(l2_init))
#        self.bn = keras.layers.BatchNormalization()
        self.out = keras.layers.Dense(2,activation='softmax', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(l2_init))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs):
        inputs = inputs.to_tensor()
        inputs = tf.expand_dims(inputs, axis=-1)
        
        # Encoder : (batch, last_channel)
        Encoder = [self.scrnn(inputs[:,idx]) for idx in self.roi_id]
        concated =tf.stack(Encoder, axis=1)
        seq_out = self.lstm(concated)
        attention_out = self.attention(seq_out)
        attended_flat = keras.layers.Flatten()(attention_out)

        dense1 = self.dense(attended_flat)
        dense1 = self.dropout(dense1)
        output = self.out(dense1)

        return output


class SC_CNN_adrnn_roi(keras.Model):
    def __init__(self, kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2_init, drop_rate, roi, **kwargs):
        super().__init__(**kwargs)

        self.roi_id = roi
        self.scrnn = layers.SCCNN_dilate(kernals=kernal_list, stride=1, padding='same', filters=filter_list, activation=cnn_activation, activation_param=activation_param, init=init, l2=l2_init, rate=drop_rate)

        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_hidden, return_sequences=True, time_major=False))

        self.attention = layers.Additive_attention(units=attention_hidden)
        self.dense = tf.keras.layers.Dense(dense_hidden, activation='relu',kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(l2_init))
#        self.bn = keras.layers.BatchNormalization()
        self.out = keras.layers.Dense(2,activation='softmax', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(l2_init))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs):
        inputs = inputs.to_tensor()
        inputs = tf.expand_dims(inputs, axis=-1)
        
        # Encoder : (batch, last_channel)
        Encoder = [self.scrnn(inputs[:,idx]) for idx in self.roi_id]
        concated =tf.stack(Encoder, axis=1)
        seq_out = self.lstm(concated)
        attention_out = self.attention(seq_out)
        attended_flat = keras.layers.Flatten()(attention_out)

        dense1 = self.dense(attended_flat)
        dense1 = self.dropout(dense1)
        output = self.out(dense1)

        return output


class SC_CNN_asrnn_roi(keras.Model):
    def __init__(self, kernal_list, filter_list, padding, cnn_activation, activation_param, lstm_hidden, attention_hidden, dense_hidden, init, l2_init, drop_rate, roi, **kwargs):
        super().__init__(**kwargs)

        self.roi_id = roi
        self.scrnn = layers.SCCNN_bn(kernals=kernal_list, stride=1, padding=padding, filters=filter_list, activation=cnn_activation, activation_param=activation_param, init=init, l2=l2_init, rate=drop_rate)

        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_hidden, return_sequences=False, time_major=False))
        self.dense1 = keras.layers.Dense(dense_hidden, activation=keras.layers.LeakyReLU(activation_param), kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2_init))
        self.out = keras.layers.Dense(2, activation='softmax', kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(l2_init))
        self.attention = layers.Additive_attention(attention_hidden)
        self.dropout = tf.keras.layers.Dropout(drop_rate)


    def call(self, inputs):
        inputs = inputs.to_tensor()
        inputs = tf.expand_dims(inputs, axis=-1)
        Encoder = [self.scrnn(inputs[:,idx]) for idx in self.roi_id]
        concated = tf.stack(Encoder,axis=1)
        
        slice_list, slice_num = self.slice(tf.transpose(concated, perm=[0,2,1]), window=10, shift=5)
        slice_output_list = list()

        for i in range(slice_num):
            slice_x = tf.transpose(slice_list[i], perm=[0,2,1])
            slice_out_all = self.bilstm(slice_x)
            slice_output_list.append(slice_out_all)

        rnn_output = tf.stack(slice_output_list, axis=1)
        attention = self.attention(rnn_output)
        attention_flat = keras.layers.Flatten()(attention)

        out_dense = self.dense1(attention_flat)
        out_dense = self.dropout(out_dense)
        output = self.out(out_dense)

        return output

    def slice(self, inputs, window, shift):
        output = list()
        if inputs.shape.as_list()[2] >= window:
            quo ,rem = divmod((inputs.shape.as_list()[2] - window), shift)
            idx =  quo + (rem !=0) + 1
            for i in range(idx):
                if i == (idx - 1):
                    output.append(inputs[:,:,-window:])
                else:
                    output.append(inputs[:,:,i*shift:i*shift+window])
        else:
            idx = 1
            output.append(inputs)
        
        return output, idx