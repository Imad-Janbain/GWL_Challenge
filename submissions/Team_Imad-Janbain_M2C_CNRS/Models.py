from sklearn.model_selection import train_test_split

import optuna
import numpy as np
import pandas as pd
import sklearn

import keras
from keras.layers import Activation

from keras_resnet.models import ResNet

from keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D, Add, AveragePooling1D
from tensorflow.keras.models import Model

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D, Add, AveragePooling1D
from tensorflow.keras.models import Model


def build_cnn_model(look_back, num_layers, num_units, kernel_size, filter_size):
  # Build the CNN model
  inputs = keras.Input(shape=(look_back, input_dim))
  x = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size)(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  for i in range(num_layers - 1):
    x = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(num_units)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  output = keras.layers.Dense(1)(x)
  model = keras.Model(inputs=inputs, outputs=output)
  return model

def build_cnn_bilstm_attention_model(look_back, num_layers, num_units, kernel_size, filter_size):
  # Build the CNN-BiLSTM-Attention model
  inputs = keras.Input(shape=(look_back, input_dim))
  x = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size)(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  for i in range(num_layers - 1):
    x = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
  x = keras.layers.Bidirectional(keras.layers.LSTM(num_units,return_sequences=True))(x)
  x = keras.layers.Bidirectional(keras.layers.LSTM(num_units))(x)
  attention = keras.layers.Attention()(x)
  x = keras.layers.Dense(num_units)(attention)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  output = keras.layers.Dense(1)(x)
  model = keras.Model(inputs=inputs, outputs=output)
  return model

def build_resnet_model(look_back, num_layers, num_units, dropout_rate, learning_rate):
  # Define the model input
  input_shape = (look_back, input_dim)
  model_input = Input(shape=input_shape)

  # Add the ResNet layers
  resnet = ResNet(model_input, num_layers, num_units, dropout_rate)

  # Add a dense layer for the output
  output = Dense(1)(resnet)

  # Create the model
  model = Model(model_input, output)

  return model



def build_wavenet_model(look_back, num_layers, num_units, dilation_rate):
  # Build the WaveNet model
  inputs = keras.Input(shape=(look_back, input_dim))
  x = keras.layers.Conv1D(filters=num_units, kernel_size=2, dilation_rate=dilation_rate)(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  for i in range(num_layers - 1):
    x = keras.layers.Conv1D(filters=num_units, kernel_size=2, dilation_rate=dilation_rate)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
  x = keras.layers.GlobalAveragePooling1D()(x)
  output = keras.layers.Dense(1)(x)
  model = keras.Model(inputs=inputs, outputs=output)
  return model

def build_transformer_model(look_back, num_layers, num_units, num_heads):
  # Build the transformer model
  inputs = keras.Input(shape=(look_back, input_dim))
  x = keras.layers.Masking()(inputs)
  for i in range(num_layers):
    x = keras.layers.MultiHeadAttention(num_heads)(x)
    x = keras.layers.Dense(num_units)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
  x = keras.layers.GlobalAveragePooling1D()(x)
  output = keras.layers.Dense(1)(x)
  model = keras.Model(inputs=inputs, outputs=output)
  return model


def build_tcn_model(look_back, num_layers, num_units, dilation_rate):
  # Build the TCN model
  inputs = keras.Input(shape=(look_back, input_dim))
  x = keras.layers.Conv1D(filters=num_units, kernel_size=2, dilation_rate=dilation_rate)(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  for i in range(num_layers - 1):
    x = keras.layers.Conv1D(filters=num_units, kernel_size=2, dilation_rate=dilation_rate)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
  x = keras.layers.GlobalAveragePooling1D()(x)
  output = keras.layers.Dense(1)(x)
  model = keras.Model(inputs=inputs, outputs=output)
  return model
