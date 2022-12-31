# Team Example

In this file the submission is described. 

## Author(s)

- Imad JANBAIN (M2C CNRS - University of Rouen) - imad.janbain@univ-rouen.fr

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [X] Sweden 1
- [X] Sweden 2
- [X] USA

## Model description

We used several models including CNN-BiLSTM-Attention, Resnet, Wavenet, TCN, Transformer.They belong to the category of deep learning models.
The models are implemented using python with the Keras-Tensorflow libraries.
All the models hyperparameters are tuning using Optuna library.

## Model workflow to reproduce
The user can select the well location and the desired model by commenting or uncommenting the necessary lines in the code. The rest of the process is automated, so the code do the rest for you :).
The code contains several functions to accomplish the task.
1- The first one is responsible for defining the training and testing periods based on the selected well location and saving the dates in a dictionary.
2- The second function prepares the data by reading the data files and using the date dictionary to generate the necessary input and target data for model training, as well as the input and target scalers.
3- The third function uses the Optuna library to optimize the model hyperparameters. The best models, input and target scalers, and best hyperparameters are saved for later use.
4- The fourth function evaluates the models on the training and validation data and saves the evaluation metrics (e.g., RMSE, MAE, MAPE) in a CSV file.
5- The final function loads the saved files and uses them to make predictions and visualize the model's performance, including a 95% confidence interval.

To facilitate tracking, the names of all the saved files include the model name and the selected well location.

## Supplementary model data used

I only changed the name of the date columns across all the data files by renaming them "date" to make the data easier to work with.
It is possible also to add some 

## Estimation of effort

I spent about 1 hour reading and organizing the data, and around 3.5 days writing and modifying the code to ensure its flexibility. This took longer due to the fact that each data file had its own unique start and end dates, requiring extra time to properly access and automate the process. I am currently in the process of tuning the models and will provide the results as soon as they are ready.

| Location    | Development time (hrs) | Calibration time (s) | Total time (hrs) | 
|-------------|------------------------|----------------------|------------------|
| Netherlands | ~                      |                      |                  |
| Germany     |                        |                      |                  |
| Sweden 1    |                        |                      |                  |
| Sweden 2    |                        |                      |                  |
| USA         |                        |                      |                  |

## Additional information

