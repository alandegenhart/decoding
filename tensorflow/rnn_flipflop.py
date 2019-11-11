# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

print('Using TensorFlow v {}'.format(tf.__version__))


# Define a function to generate associated input and output samples
def gen_training_ex(n_inputs, seq_len, n_seq):
  p = .2  # Probability of an event

  X = np.zeros((n_seq, seq_len, n_inputs))
  Y = np.zeros((n_seq, seq_len, n_inputs))

  for seq in range(n_seq):
      for i in range(n_inputs):
        # Generate events from a binomial distribution
        x_pos = np.random.binomial(1, p/2, (1, seq_len))
        x_neg = np.random.binomial(1, p/2, (1, seq_len)) * -1
        x = x_pos + x_neg

        # Determine associated output for the input
        y = np.zeros(x.shape)
        y[0, 1] = x[0, 1]  # Initialize output

        for j in range(1, seq_len):
          # If the input is the same as the previous output, set the output to the
          # previous value
          if (x[0,j] == y[0,j-1]) | (x[0,j] == 0.):
            y[0,j] = y[0,j-1]
          else:
            y[0,j] = x[0,j]

        # Add to output matrices
        X[seq, :, i] = x
        Y[seq, :, i] = y

  return X, Y


# Function to plot data
def plot_training_ex(X, Y, plot_name):
  # Create figure
  plt.figure()
  plt.xlabel('Timestep')
  plt.ylabel('Input/Output')

  # Loop over input/output pairs and plot
  n_ex = X.shape[0]
  for i in range(n_ex):
    x = X[i,:]
    y = Y[i,:]

    # Scale y-axis values to be between i - .4 and i + .4
    x = (x * 0.4) + i + 1
    y = (y * 0.4) + i + 1
    plt.plot(x, color = 'black')
    plt.plot(y, color = 'red')

  plt.yticks(np.arange(n_ex) + 1)
  plt.savefig('results/FlipFlopRNN_{}.pdf'.format(plot_name))


# Build the TF model
def build_model(n_inputs, n_rnn, seq_len):
  model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(
        n_rnn,
        batch_input_shape=[None, seq_len, n_inputs],  # Here, 'None' means that the number of sequences is not specified aprioi
        return_sequences=True),  # Return the entire sequence (not just the last state)
    tf.keras.layers.Dense(n_inputs)
  ])

  # Compile the model
  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mae')

  return model


# Define a callback function that will get called each epoch
class PrintDot(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('\n')  # Every 100 lines, print a new line
    print('.', end='')


def main():
    # Define system parameters and generate training data
    n_inputs = 3
    n_ts = 100  # Number of steps per sequence
    n_seq = 10  # Number of sequences to generate
    X, Y = gen_training_ex(n_inputs, n_ts, n_seq)
    plot_training_ex(X[0, :, :].T, Y[0, :, :].T, 'ExampleTrainingData')
    print('Data shape: {}'.format(X.shape))

    # Build the model
    n_rnn = 128
    model = build_model(n_inputs, n_rnn, n_ts)
    model.build()
    model.summary()

    # Now try running some test data through to get the output
    test_input = np.expand_dims(X.T, axis = 1)
    Y_predict = model.predict(X)
    print('Predicted data shape: {}'.format(Y_predict.shape))
    plot_training_ex(X[0, :, :].T, Y_predict[0, :, :].T, 'PreTraining')

    EPOCHS = 50

    history = model.fit(
      X, Y,
      epochs=EPOCHS,
      validation_split=0.2,
      verbose=0,
      callbacks=[PrintDot()])

     # Now predict the input
    Y_predict = model.predict(X)
    plot_training_ex(X[0, :, :].T, Y_predict[0, :, :].T, 'TestData_100samp')

    # This doesn't do great.  What if we try more data?
    n_seq = 200
    X_large, Y_large = gen_training_ex(n_inputs, n_ts, n_seq)

    history = model.fit(
      X_large, Y_large,
      epochs=EPOCHS, validation_split=0.2, verbose=0,
      callbacks=[PrintDot()])

    # Now predict the input
    Y_predict = model.predict(X_large)
    plot_training_ex(
        X_large[-1, :, :].T,
        Y_large[-1, :, :].T,
        'TestData_200seq')

    # Finally, to test generalizability, generate some new (previously
    # un-observed) data and predict
    X_new, Y_new = gen_training_ex(n_inputs, n_ts, 10)
    Y_new_predict = model.predict(X_new)
    plot_training_ex(
        X_new[0, :, :].T,
        Y_new[0, :, :].T,
        'TestData_NewData')

    return None


# Run main function
main()
