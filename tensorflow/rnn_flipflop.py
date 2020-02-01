# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import plottools as pt

#print('Using TensorFlow v {}'.format(tf.__version__))


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
def plot_training_ex(X, Y_act, Y_pred, plot_title, axh=None):
	"""Plot input/output data for flip-flop model."""

	# Create figure if an axis handle is not provided
	if axh is None:
		fh, axh = pt.create_subplot(n_row=1, n_col=1)
		axh = axh[0]
	else:
		fh = None

	# Set axis labels
	axh.set_xlabel('Timestep')
	axh.set_ylabel('Input/Output')
	axh.set_title(plot_title)

	# Loop over input/output pairs and plot
	n_ex = X.shape[0]
	for i in range(n_ex):
		x = X[i, :]
		y_act = Y_act[i, :]
		y_pred = Y_pred[i, :]

		# Scale y-axis values to be between i - .4 and i + .4
		x = (x * 0.4) + i + 1
		y_act = (y_act * 0.4) + i + 1
		y_pred = (y_pred * 0.4) + i + 1

		axh.plot(x, color = 'black')
		axh.plot(y_act, color = 'red')
		axh.plot(y_pred, color = 'orange')

	axh.set_yticks(np.arange(n_ex) + 1)

	return fh


def plot_history(history, axh=None):
	"""Plot model training history."""
	
	# Create figure if an axis handle is not provided
	if axh is None:
		fh, axh = pt.create_subplot(n_row=1, n_col=1)
		axh = axh[0]
	else:
		fh = None

	# Plot training error and validation error
	loss_h, = axh.plot(history.history['loss'])
	val_loss_h, = axh.plot(history.history['val_loss'])

	# Add axis labels and legend
	axh.set_xlabel('Epoch')
	axh.set_ylabel('MSE')
	axh.legend((loss_h, val_loss_h), ('Training error', 'Validation error'))
	axh.set_title('Training history')

	return fh


def build_model(n_inputs, n_rnn=128, use_regularizer=False, reg_l=0.01):
	"""Build RNN model using the 'Sequential' class.

	The simple RNN model has two layers - a SimpleRNN layer and a dense output
	layer.

	The input shape is defined in the form [n_batches, n_timesteps, n_inputs].
	In order to allow arbitrary-length sequences to be predicted, the batch shape
	should be [None, None, n_inputs].

	A note on regularization -- there are two different regularizers that can
	be applied.  The 'kernel' regularizer is applied to the weights that map the
	inputs to the recurrent units, and the 'recurrent kernel' are the weights
	of the recurrent units.
	"""
	# Determine if regularization is to be used
	if use_regularizer:
		reg = tf.keras.regularizers.l2(reg_l)
	else:
		reg = None

	# Define the model.
	model = tf.keras.Sequential([
		tf.keras.layers.SimpleRNN(
		    n_rnn,
		    batch_input_shape=[None, None, n_inputs],
		    return_sequences=True,
		    kernel_regularizer=reg,
		    recurrent_regularizer=reg
			),
		tf.keras.layers.Dense(n_inputs)
	])

	# Compile the model
	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss='mse')

	return model


# Define a callback function that will get called each epoch
class PrintDot(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
	    if epoch % 100 == 0: print('\n')  # Every 100 lines, print a new line
	    print('.', end='')


def pca_analysis(model, X):
	"""Run PCA analysis on model to visualize hidden layer activity.

	To get the values of the intermediate layer, a new model needs to be 
	created. This model takes the normal input from the RNN, and returns the
	output of the rnn layer.  The values of the hidden layer can then be found
	using the 'predict()' method.
	"""

	# --- Get hidden layer activations and run PCA ---

	# Create new model and predict input
	inputs = model.input
	outputs = [model.layers[0].output, model.layers[1].output]
	act_model = tf.keras.Model(inputs=inputs, outputs=outputs)
	activations = act_model.predict(X)

	# Format activations into dimensions x observations
	# Input dimensions: (obs x time x units)
	n_inputs = X.shape[2]
	n_rnn = activations[0].shape[2]
	A = np.transpose(activations[0], (2, 0, 1))  # Now (units x obs x time)
	Y = np.transpose(activations[1], (2, 0, 1))
	A = A.reshape((n_rnn, -1), order='C')  # Now (units x (obs*time))
	Y = Y.reshape((n_inputs, -1), order='C')

	# Run PCA (note that this is actually PPCA)
	from sklearn.decomposition import PCA
	pca = PCA(n_components=20)
	pca.fit(A.T)
	Z = pca.transform(A.T)

	# --- Plot results ---

	# Create colormap for points.  Use an RGB map of the output
	color = np.copy(Y.T)
	color[color < -1] = -1
	color[color > 1] = 1
	color = (color + 1)/2  # Colors must be between 0 and 1

	# Figure 1: Variance explained and 2D projections

	# Setup figure
	fh, ax_h = pt.create_subplot(n_row=1, n_col=4)

	# Subplot 1: Plot fraction of variance explained
	idx = 0
	dims = np.arange(pca.n_components_) + 1
	ax_h[idx].plot(dims, pca.explained_variance_ratio_, color='k', marker='.')
	ax_h[idx].set_xlabel('PCA dim.')
	ax_h[idx].set_ylabel('Fraction of variance explained')
	ax_h[idx].set_title('Fraction of variance explained')

	# Subplots 2-4: 2D projections
	# Iterate over dimensions and plot
	plot_dim = [[0, 1], [0, 2], [1, 2]]
	for ax, d in zip(ax_h[1:4], plot_dim):
	    ax.scatter(Z[:, d[0]], Z[:, d[1]], marker='.', c=color)
	    ax.set_xlabel('PC {}'.format(d[0] + 1))
	    ax.set_ylabel('PC {}'.format(d[1] + 1))
	    ax.set_title('Dim {} vs Dim {}'.format(d[0] + 1, d[1] + 1))

	# Save figure
	fh.savefig('results/FlipFlopRNN_{}.pdf'.format('PCA'))

	# Figure 2: 3d representation
	# Note that this needs to be done separately from the above plots
	# because a separate argument is required when creating a subplot
	# for a 3d plot

	# This import registers the 3D projection, but is otherwise unused.
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

	# Create figure
	fh = plt.figure()
	ax = fh.add_subplot(111, projection='3d')

	# Plot
	ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], marker='o', c=color)
	ax.set_xlabel('PC 1')
	ax.set_ylabel('PC 2')
	ax.set_zlabel('PC 3')
	ax.set_title('Top 3 PCs')

	# Save figure
	fh.savefig('results/FlipFlopRNN_{}.pdf'.format('PCA_3D'))

	return None

