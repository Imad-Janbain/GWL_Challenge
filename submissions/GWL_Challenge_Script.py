
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

# the remaining models are in the Models file.. you can load and try them .. please uncomment the line bellow with the correspondent selected model_name in the script..
# from Models import build_cnn_model, build_cnn_bilstm_attention_model, build_resnet_model, build_wavenet_model, build_transformer_model, build_tcn_model


# Here we have to define the necessary configuration for each well station:
# here you need just to uncomment only the selected well location.. and the code do the rest :)

Well_Location = 'Germany'
# Well_Location = 'Netherlands' 
# Well_Location = 'USA' 
# Well_Location = 'Sweden_1' 
# Well_Location = 'Swedem_2' 


model_name = 'cnn'
# model_name = 'tcn'
# model_name = 'transformer'
# model_name = 'cnn-bilstm-attention'
# model_name = 'resnet'
# model_name = 'wavenet'





Data_Path = 'data/' + Well_Location + '/'
input_file = Data_Path + 'input_data.csv'
target_file = Data_Path + 'heads.csv'


if Well_Location == 'USA':
    input_dim = 5 # 5 original columns + N in case we will add N rolling columns for the precipitation

else:
    input_dim = 9 # 9 original columns + N in case we will add N rolling columns for the precipitation

# we can use the simple function below to make the model flexible for each well location, we can add more well for future challenge..
def get_periods(well_station):
  if well_station == 'Germany':
    dates = {
      'train_start_date': '2002-05-01',
      'train_end_date': '2016-12-31',
      'test_start_date': '2017-01-01',
      'test_end_date': '2021-12-31'
    }
  elif well_station == 'Netherlands':
    dates = {
      'train_start_date': '2000-01-01',
      'train_end_date': '2015-09-10',
      'test_start_date': '2016-01-01',
      'test_end_date': '2021-12-31'
    }

  elif well_station == 'USA':
    dates = {
      'train_start_date': '2002-03-01',
      'train_end_date': '2016-12-31',
      'test_start_date': '2017-01-01',
      'test_end_date': '2022-05-31'
    }
  elif well_station == 'Sweden_1':
    dates = {
      'train_start_date': '2001-01-01',
      'train_end_date': '2015-12-31',
      'test_start_date': '2016-01-01',
      'test_end_date': '2021-12-31'
    }

  elif well_station == 'Sweden_2':
    dates = {
      'train_start_date': '2001-01-01',
      'train_end_date': '2015-12-31',
      'test_start_date': '2016-01-01',
      'test_end_date': '2021-12-31'
    }            
  return dates


dates = get_periods(Well_Location)

n_output = 1 # here we have just one target..
validation_split = 0.2
            

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


# the function below is dedicated to prepare the data in a suitable form to train the deep learning model, it will use the predifined configs at the biginning of the code
# like the dates distionary.. 
# Note that i adjusted manually the date columns of all the file to be "date"
def prepare_data(input_file, target_file, dates, look_back, validation_split):
    # Read the input and target files
    input_df = pd.read_csv(input_file)
    target_df = pd.read_csv(target_file)

    # Merge the input and target data
    df = pd.merge(input_df, target_df, on='date')

    # Split the data into training and testing sets
    train_df = df[(df['date'] >= dates['train_start_date']) & (df['date'] <= dates['train_end_date'])]
    test_df = df[(df['date'] >= dates['test_start_date']) & (df['date'] <= dates['test_end_date'])]

    # Standardize the data using scalers
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Create input sequences using the look-back value
    X_train = []
    y_train = []
    for i in range(look_back, train_df.shape[0]):
        X_train.append(input_scaler.fit_transform(train_df.iloc[i-look_back:i, :].drop(columns=['date', 'head'])))
        y_train.append(target_scaler.fit_transform(train_df[i:i+1][['head']]))
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Split the training data into a training set and a validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split)

    # Create input sequences using the look-back value
    X_test = []
    y_test = []
    for i in range(look_back, test_df.shape[0]):
        X_test.append(input_scaler.transform(test_df.iloc[i-look_back:i, :].drop(columns=['date', 'head'])))
        y_test.append(target_scaler.transform(test_df[i:i+1][['head']]))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Return the prepared data and scalers
    return X_train, y_train, X_val, y_val, X_test, y_test, input_scaler, target_scaler



import numpy as np
import matplotlib.pyplot as plt


def predict_and_plot(model, model_name, well_station_name, input_scaler, target_scaler, X_train, y_train, X_val, y_val, X_test, y_test, dates, best_hyperparams):
    # Make predictions on the training, validation, and test periods
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    # Reshape predictions and target data to be suitable for plotting
    train_predictions_reshaped = train_predictions.reshape((train_predictions.shape[0], n_output))
    val_predictions_reshaped = val_predictions.reshape((val_predictions.shape[0], n_output))
    test_predictions_reshaped = test_predictions.reshape((test_predictions.shape[0], n_output))
    y_train_reshaped = y_train.reshape((y_train.shape[0], n_output))
    y_val_reshaped = y_val.reshape((y_val.shape[0], n_output))
    y_test_reshaped = y_test.reshape((y_test.shape[0], n_output))

    # Unscale predictions and target data
    train_predictions_unscaled = target_scaler.inverse_transform(train_predictions_reshaped)
    val_predictions_unscaled = target_scaler.inverse_transform(val_predictions_reshaped)
    test_predictions_unscaled = target_scaler.inverse_transform(test_predictions_reshaped)
    y_train_unscaled = target_scaler.inverse_transform(y_train_reshaped)
    y_val_unscaled = target_scaler.inverse_transform(y_val_reshaped)
    y_test_unscaled = target_scaler.inverse_transform(y_test_reshaped)

    # Convert predictions and target data to dataframes
    train_predictions_df = pd.DataFrame(train_predictions_unscaled, index=dates[:len(train_predictions_unscaled)], columns=target_columns)
    val_predictions_df = pd.DataFrame(val_predictions_unscaled, index=dates[len(train_predictions_unscaled):len(train_predictions_unscaled)+len(val_predictions_unscaled)], columns=target_columns)
    test_predictions_df = pd.DataFrame(test_predictions_unscaled, index=dates[len(train_predictions_unscaled)+len(val_predictions_unscaled):], columns=target_columns)
    y_train_df = pd.DataFrame(y_train_unscaled, index=dates[:len(train_predictions_unscaled)], columns=target_columns)
    y_val_df = pd.DataFrame(y_val_unscaled, index=dates[len(train_predictions_unscaled):len(train_predictions_unscaled)+len(val_predictions_unscaled)], columns=target_columns)
    y_test_df = pd.DataFrame(y_test_unscaled, index=dates[len(train_predictions_unscaled)+len(val_predictions_unscaled):], columns=target_columns)


    # Concatenate predictions on the training, validation, and test periods
    predictions_df = pd.concat([train_predictions_df, val_predictions_df, test_predictions_df])
    y_df = pd.concat([y_train_df, y_val_df, y_test_df])

    # Calculate the 95% confidence interval for the predictions
    predictions_ci = predictions_df.apply(lambda x: x.quantile(q=[0.025, 0.975]))

    # Plot the model predictions along with the original target data and the confidence interval
    ax = y_df.plot(figsize=(12,6), color='black', title=f'{model_name} Prediction Results for {well_station_name}')
    predictions_df.plot(color='red', alpha=0.6, ax=ax)
    predictions_ci.plot(color='red', alpha=0.2, ax=ax)
    ax.fill_between(predictions_ci.index, predictions_ci.iloc[:,0], predictions_ci.iloc[:,1], color='red', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.set_title(f'{model_name} Prediction Results for {well_station_name}')
    plt.show()

    # Save the prediction results to a CSV file
    predictions_ci.to_csv(f'{model_name}_{well_station_name}_prediction_results.csv', index=True)


# Define the objective function
def objective(trial):
      
    # Get the model hyperparameters
    look_back = trial.suggest_int('look_back', 1, 50)
    batch_size = trial.suggest_int('batch_size', 8, 128)

    if model_name == 'cnn':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      kernel_size = trial.suggest_int('kernel_size', 2, 5)
      filter_size = trial.suggest_int('filter_size', 3, 7)

      model = build_cnn_model(look_back, num_layers, num_units, kernel_size, filter_size)

    elif model_name == 'lstm':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      model = build_lstm_model(look_back, num_layers, num_units, input_dim)

    elif model_name == 'bilstm':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      model = build_bilstm_model(look_back, num_layers, num_units, input_dim)


    elif model_name == 'cnn-lstm':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      kernel_size = trial.suggest_int('kernel_size', 2, 5)
      model = build_cnn_lstm_model(look_back, num_layers, num_units, kernel_size, input_dim)


    elif model_name == 'cnn-bilstm-attention':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      kernel_size = trial.suggest_int('kernel_size', 2, 5)
      model = build_cnn_bilstm_attention_model(look_back, num_layers, num_units, kernel_size, input_dim)


    elif model_name == 'bilstm-attention':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      model = build_bilstm_attention_model(look_back, num_layers, num_units, input_dim)
    
    elif model_name == 'resnet':
        
        num_layers = trial.suggest_int('num_layers', 1, 3)
        num_units = trial.suggest_int('num_units', 8, 64)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        model = build_resnet_model(look_back, num_layers, num_units, dropout_rate, learning_rate)
    
    elif model_name == 'wavenet':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      dilation_rate = trial.suggest_int('dilation_rate', 1, 2)
      model = build_wavenet_model(look_back, num_layers, num_units, dilation_rate)


    elif model_name == 'tcn':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      dilation_rate = trial.suggest_int('dilation_rate', 1, 2)
      model = build_tcn_model(look_back, num_layers, num_units, dilation_rate)

    elif model_name == 'transformer':
      num_layers = trial.suggest_int('num_layers', 1, 3)
      num_units = trial.suggest_int('num_units', 8, 64)
      num_heads = trial.suggest_int('num_heads', 2, 8)

      model = build_transformer_model(look_back, num_layers, num_units, num_heads)

    # Prepare the data
    X_train, y_train, X_val, y_val, X_test, y_test, input_scaler, target_scaler = prepare_data( input_file, target_file, dates, look_back, validation_split)
    
    input_dim = X_train.shape[2]

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])

    # Evaluate the model on the test set
    loss = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=False)

    return loss

  # Create a study
study = optuna.create_study()

  # Set up the study
study.optimize(objective, n_trials=50)

# Get the best trial from the Optuna optimization
best_trial = study.best_trial

# Extract the best model, scalers, and hyperparameters from the best trial
best_model = best_trial.user_attrs['model']
best_input_scaler = best_trial.user_attrs['input_scaler']
best_target_scaler = best_trial.user_attrs['target_scaler']
best_hyperparameters = best_trial.user_attrs['hyperparameters']


# Save the model, input scaler, target scaler, and best hyperparameters
import pickle

model_file = f"{model_name}_{well_location}_model.pkl"
input_scaler_file = f"{model_name}_{well_location}_input_scaler.pkl"
target_scaler_file = f"{model_name}_{well_location}_target_scaler.pkl"
hyperparameters_file = f"{model_name}_{well_location}_hyperparameters.pkl"

with open(model_file, "wb") as f:
  pickle.dump(best_model, f)

with open(input_scaler_file, "wb") as f:
  pickle.dump(best_input_scaler, f)

with open(target_scaler_file, "wb") as f:
  pickle.dump(best_target_scaler, f)

with open(hyperparameters_file, "wb") as f:
  pickle.dump(best_hyperparameters, f)




# Load the model, input scaler, target scaler, and best hyperparameters
import pickle

model_file = f"{model_name}_{well_location}_model.pkl"
input_scaler_file = f"{model_name}_{well_location}_input_scaler.pkl"
target_scaler_file = f"{model_name}_{well_location}_target_scaler.pkl"
hyperparameters_file = f"{model_name}_{well_location}_hyperparameters.pkl"

with open(model_file, "rb") as f:
  model = pickle.load(f)

with open(input_scaler_file, "rb") as f:
  input_scaler = pickle.load(f)

with open(target_scaler_file, "rb") as f:
  target_scaler = pickle.load(f)

with open(hyperparameters_file, "rb") as f:
  best_hyperparameters = pickle.load(f)


X_train, y_train, X_val, y_val, X_test, y_test, input_scaler, target_scaler = prepare_data(
      input_file, target_file, dates, best_hyperparameters['look_back'], validation_split)

predict_and_plot(model, model_name, well_station_name, input_scaler, target_scaler, X_train, y_train, X_val, y_val, X_test, y_test, dates, best_hyperparams)


# Evaluate the model on the training and validation periods
train_evaluation = model.evaluate(X_train, y_train, verbose=0)
val_evaluation = model.evaluate(X_val, y_val, verbose=0)

    # Create a dictionary of evaluation metrics
evaluation_metrics = {
        'Metric': ['RMSE', 'MSE', 'MAE', 'MAPE'],
        'Train': [np.sqrt(train_evaluation[1]), train_evaluation[1], train_evaluation[2], np.mean(np.abs((y_train_unscaled - train_predictions_unscaled) / y_train_unscaled)) * 100],
        'Validation': [np.sqrt(val_evaluation[1]), val_evaluation[1], val_evaluation[2], np.mean(np.abs((y_val_unscaled - val_predictions_unscaled) / y_val_unscaled)) * 100]
    }

    # Convert the dictionary to a Pandas DataFrame
evaluation_df = pd.DataFrame(evaluation_metrics)

    # Save the evaluation results to a CSV file
evaluation_df.to_csv(f'{model_name}_{well_station_name}_evaluation_results.csv', index=False)
