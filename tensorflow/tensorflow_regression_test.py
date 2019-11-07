# Import libraries

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print('Tensorflow version: {}'.format(tf.__version__))

# Download the dataset.  The dataset used for this exercise is a database of
# auto fuel efficiency and characteristics (weight, number of cylinders, etc.).
# This will download the data to the local Keras directory.
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# Load data using Pandas.  The dataset is a CSV file.
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = '?',
                          comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.head())  # Plots the first 5 rows of the data in tabular format
print(dataset.tail())  # Plots the last 5 rows of the data in tabular format

# Check to see which rows have missing data
dataset.isna()  # Will show all rows
print(dataset.isna().sum())  # Provides the sum across columns
dataset = dataset.dropna()  # Drop observations

# Remove the 'Origin' column.  The 'pop' method returns the specified field from
# the dataset.  This field is removed from the original dataset.
origin = dataset.pop('Origin')

# Add new entries to the dataset
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

# Display updated dataset
print(dataset.tail())

# Split data into training and test sets.
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

# Plot the raw data.  The 'seaborn' module provides a simple way of doing this
# using the 'pairplot' function.
g = sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']],
             diag_kind = 'kde')
# Save
fig_name = 'RawData'
g.savefig('results/{}.pdf'.format(fig_name))
# The 'diag_kind' input describes the type of diagonal plot.  Options are 'auto'
# 'hist', or 'kde'.  'kde': 'kernel density estimate'.

# Look at dataset in more detail
train_stats = train_dataset.describe()
train_stats.pop('MPG')  # Remove MPG, as we don't need to describe this
print(train_stats)

train_stats = train_stats.transpose()
print(train_stats)  # Note that the print() and display elements are formatted differently

#-------------------------------------------------------------------------------
# Model training

# Split data into test and train sets
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Define a function to calculate the norm of an input variable. Note that
# variable scope in Python follows a 'LEGB' ('local', 'extended', 'global',
# 'built-in') scope. Variables will first be looked at locally, and then in the
# extended scope. In this case, 'train_stats' is not a local variable, but it
# can still be accessed in the extended scope.
def norm(x):
  norm_x = (x - train_stats['mean']) / train_stats['std']
  return norm_x  # More 'Pythonic' to do in a single line, but whatever...


norm_train_data = norm(train_dataset)
norm_test_data = norm(test_dataset)  # This is OK b/c it uses the 'train' stats

print(type(norm_train_data))

# Define a function to build the model
def build_model():
  model = keras.Sequential([
      layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation=tf.nn.relu),
      layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer = optimizer,
                metrics = ['mean_absolute_error', 'mean_squared_error'])

  return model

# Build the model -- note that this appears to randomly initialize the weights
model = build_model()
# Display a summary of the model
model.summary()

# Test to make sure that the model works properly. Do this by running the model
# on some test data.  This should work b/c the weights are randomly initialized.
# NOTE -- this is currently generating an warning in tensorflow -- it appears
# that there isn't an appropriate adapter for DataFrame (pandas) data types
# for version 2.0.  To fix this, use 'DataFrame.values' to get a np array
example_batch = norm_train_data[:10].values  # Use the first 10 samples
example_result = model.predict(example_batch)
print(example_result)

# Train the model.  Before doing this, define a callback class/function to
# display a dot every iteration

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('\n')  # Every 100 lines, print a new line
    print('.', end='')


EPOCHS = 1000

history = model.fit(
  norm_train_data.values, train_labels.values,
  epochs=EPOCHS, validation_split=0.2, verbose=0,
  callbacks=[PrintDot()])

# Now that the model is trained, visualize the training progress.  This will be
# in the 'history' output of the fitting function.
# Display the training history
hist = pd.DataFrame(history.history)  # Convert
hist['epoch'] = history.epoch
hist.tail()

# Plot the training history.  First define a function to do this

def plot_history(history, save_name):
  # Convert the TF history object to a Pandas data frame
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch  # Add training epoch

  # Create figure -- absolute error
  plt.figure(figsize=(10,5))
  plt.subplot(1, 2, 1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean absolute error (mpg)')
  plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
  plt.ylim([0,5])
  plt.legend()

  # Create figure -- mean-squared error
  plt.subplot(1, 2, 2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean squared error (mpg)')
  plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
  plt.ylim([0,20])
  plt.legend()

  # Show figures
  plt.savefig('results/{}.pdf'.format(save_name))


# Plot history for the trained model
plot_history(history, 'TrainResults')

# Setup a new model that stops training after ~100 iterations
model_2 = build_model()

# Setup callback function
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Run model
history_2 = model_2.fit(norm_train_data.values, train_labels.values,
                      epochs=EPOCHS,
                      validation_split = 0.2,
                      verbose = 0,
                      callbacks = [early_stop, PrintDot()])

plot_history(history_2, 'TrainResults_EarlyStopping')

# Now predict held-out data given the second model
loss, mae, mse = model_2.evaluate(norm_test_data.values, test_labels.values, verbose = 0)
print('Testing set mean absolute error: {:5.3} MPG'.format(mae))

# Predict held-out data and compare to the ground truth
test_predictions = model_2.predict(norm_test_data.values)
test_predictions = test_predictions.flatten()  # Make 1D

# Create figure
plt.figure(figsize = (5, 5))

# Scatter predicted vs actual MPG and create axis labels
plt.scatter(test_labels, test_predictions)
plt.xlabel('True MPG')
plt.ylabel('Predicted MPG')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])

# Plot unity line
plt.plot([-100, 100], [-100, 100])  # The underscore ('_') supresses output

# Save figure
plt.savefig('results/PredictionError.pdf')

# Finally, plot a histogram of prediction error
error = test_predictions - test_labels

# Create figure
plt.figure(figsize = (5, 5))
plt.hist(error, bins = 25)
plt.xlabel('Prediction error (MPG)')
plt.ylabel('Count')

# Save figure
plt.savefig('results/PredictionError_Hist.pdf')
