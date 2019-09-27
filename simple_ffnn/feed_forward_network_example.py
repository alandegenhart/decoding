# Simple feed forward neural network example

# Import libraries
import numpy as np


# Initialize network
def init_network(num_layers, num_inputs, num_units, act_fn):

    # Update num_units with input dimensionality

    # For each layer, create a dict with the parameters
    nn_mdl = []
    for l in range(num_layers):
        layer = {}
        layer['layerID'] = l
        layer['size'] = num_units[l]
        layer['act_fn'] = act_fn[l]

        if l == 0:
            input_dim = num_inputs
        else:
            input_dim = num_units[l-1]

        # Initialize weights
        W = np.random.normal(0, 0.5, (input_dim, num_units[l]))
        b = np.random.normal(0, 0.5, (num_units[l], 1))
        layer['W'] = W
        layer['b'] = b
        layer['J'] = None
        layer['z'] = None
        layer['y'] = None
        layer['de_dw'] = None
        layer['de_db'] = None

        # Add layer to model
        nn_mdl.append(layer)

    return nn_mdl


# Run network forward
def forward_pass(mdl, x):

    # Iterate over layers, compute the value of each node
    num_layers = len(mdl)
    num_outputs = mdl[num_layers - 1]['size']
    num_samples = x.shape[1]

    # Initialize state output.  This will contain the value of each state (internal and external) for each sample
    states = {'y': [], 'z': []}
    for l in range(num_layers):
        states['y'].append(np.zeros((mdl[l]['size'], num_samples)))
        states['z'].append(np.zeros((mdl[l]['size'], num_samples)))

    # Iterate through samples and run network
    for i in range(num_samples):
        for l in range(num_layers):
            # Determine input
            if l == 0:
                # If the first layer, get the input to the network
                y_in = x[:, i]

                # Make sure that y_in is a column vector
                if len(y_in.shape) == 1:
                    y_in = np.expand_dims(y_in, 1)
            else:
                # If not the first layer, get the output from the previous layer
                y_in = mdl[l-1]['y']

            # Predict z
            W = mdl[l]['W']
            b = mdl[l]['b']
            z = np.matmul(W.transpose(), y_in) + b

            # Run through activation function
            if mdl[l]['act_fn'] == 'tanh':
                y = np.tanh(z)
            elif mdl[l]['act_fn'] == 'linear':
                y = z
            else:
                y = []

            mdl[l]['z'] = z
            mdl[l]['y'] = y
            states['z'][l][:, i] = np.squeeze(z)
            states['y'][l][:, i] = np.squeeze(y)

    y_out = states['y'][num_layers - 1]  # This will get the output of the last layer

    return mdl, y_out, states


# Cost function
def cost_fun(y, y_est):
    err = (1/len(y)) * sum(np.linalg.norm(y_est - y, 2, axis=0)**2)
    return err


# Run backward pass, compute gradients and update weights
def backward_pass(mdl, x, y, y_pred, states):
    # Define training parameters
    eta = 0.1

    # To run backprop, we move backwards over the network and compute the partial derivatives
    num_layers = len(mdl)
    num_samples = y_pred.shape[1]

    # The overall updates for the various parameters are going to be a weighted sum of the contributions for each
    # observation.  To do this, we need to keep track of the partial derivatives/Jacobian for each observation as we
    # move backwards through the network.

    # Compute error w.r.t the output -- start with the last layer.  Compute the error for all samples in the batch.
    de_dy_prev_all = 2 * (y_pred - y)  # This will be a matrix of size num outputs x num samples

    # Iterate over samples and compute the gradient
    for n in range(num_samples):
        # For each sample, run backprop and compute the gradient wrt the parameters.
        de_dy_prev = np.transpose(de_dy_prev_all[:, [n]])  # 1 x num_outputs

        for l in reversed(range(num_layers)):
            # Get the relevant states
            n_l = mdl[l]['size']  # Number of elements in the current layer
            y_l = states['y'][l][:, n]  # n_l * 1
            z_l = states['z'][l][:, n]  # n_l * 1

            # First, compute the intermediates that go into the weight calculation

            # Compute the derivative of the weights in the current layer w.r.t. the output of the layer
            if mdl[l]['act_fn'] == 'linear':
                # Linear activation function, derivative is 1
                dy_dz = np.ones((n_l, 1))
            elif mdl[l]['act_fn'] == 'tanh':
                dy_dz = 1 - (np.tanh(z_l))**2

            # dy/dz is a matrix, as both y and z are vectors.  However, b/c y_i only depends on z_i (and not z_j), this
            # is just a diagonal matrix
            dy_dz = np.diag(dy_dz)
            de_dz = de_dy_prev @ dy_dz

            # Compute updates for weight terms
            # First, get the value of the next state
            if l != 0:
                y_next = states['y'][l-1][:, [n]]  # This is the state of the previous layer in the network
            else:
                y_next = x[:, [n]]  # If this is the first layer, use the input state

            dz_dw = np.tile(np.transpose(y_next), (n_l, 1))  # n_l x n_l_next
            de_dw = np.diag(de_dz) @ dz_dw  # Note - need to diagionalize de_dz

            dz_db = np.eye(n_l)
            de_db = de_dz @ dz_db

            # Add updated gradients to model
            if n == 0:
                mdl[l]['de_dw'] = de_dw
                mdl[l]['de_db'] = de_db
            else:
                mdl[l]['de_dw'] = mdl[l]['de_dw'] + de_dw
                mdl[l]['de_db'] = mdl[l]['de_db'] + de_db

            # Calculate the updated Jacobian
            de_dy_prev = de_dz @ np.transpose(mdl[l]['W'])

    # Now that the gradient has been computed for each layer, loop back over layers and update weights
    for l in range(num_layers):
        mdl[l]['W'] = mdl[l]['W'] - eta * mdl[l]['de_dw'].transpose() / num_samples
        mdl[l]['b'] = mdl[l]['b'] - eta * np.expand_dims(mdl[l]['de_db'], axis=1) / num_samples

    return mdl


# Train network
def train_network(mdl, x, y):
    # Split train and test data (maybe do this outside of this function?)

    # Determine dimensionality
    num_epochs = 200
    num_samp = x.shape[1]
    batch_size = 100
    num_batches = int(np.floor(num_samp / batch_size))

    # Iterate over epochs
    cost_val = np.zeros((1, num_epochs))
    for e in range(num_epochs):
        # Generate random indices to permute training data
        perm_idx = np.random.permutation(num_samp)

        # Iterate over batches
        for b in range(num_batches):
            # Get start and end of batch
            batch_idx_st = b * batch_size
            batch_idx_end = batch_idx_st + batch_size

            # Handle case where the batch end index is greater than the number of samples
            if batch_idx_end > num_samp:
                batch_idx_end = num_samp

            # Run forward pass - this is needed in order to calculate the internal state of the network
            batch_inds = perm_idx[batch_idx_st:batch_idx_end]
            mdl, y_pred, states = forward_pass(mdl, x[:, batch_inds])

            # Backward pass (to update weights)
            mdl = backward_pass(mdl, x[:, batch_inds], y[:, batch_inds], y_pred, states)

        # At the end of the epoch, re-calculate the loss function
        mdl, y_pred, states = forward_pass(mdl, x)
        cost_val[0, e] = cost_fun(y, y_pred)

    return mdl, cost_val


# Plot results
def plot_results():
    # Plot value of objective function
    # Plot prediction vs actual for training data
    # Plot prediction vs actual for test data

    return None


# Plot training data
def plot_training_data(data, fig_name='Simulated Data'):
    # Plot "ground truth" function values as well as simulated data
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # Setup figure
    fig = plt.figure(figsize=[5.0, 5.0])
    ax = fig.add_subplot(111)

    # Plot data
    ax.scatter(data['x'], data['y'], color='r', label='Data')
    ax.plot(data['x_gt'], data['y_gt'], color='k', label='True function')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Format plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simulated data')
    plt.legend(frameon=False)

    # Show plot and save
    plt.savefig('results/{}.pdf'.format(fig_name))

    return None


# Generate data
def generate_data(num_obs):
    # Generate a number of samples from some known (non-linear) function
    # Input:
    #   num_obs  -  number of observations to generate

    # Define parameters of the function.  For now, use a linear function with additive noise
    m = 5.5
    b = 1.2
    sig_n = 0.3

    # Draw random x-values, calculate function output, add noise
    x = np.random.normal(size=(1, num_obs))
    # y = m * x**2 + b
    y = np.tanh(x) + b
    y = y + np.random.normal(0, sig_n, y.shape)

    # Define a "ground truth" dataset over a
    x_gt = np.linspace(-5, 5, 100)
    # y_gt = m * x_gt**2 + b
    y_gt = np.tanh(x_gt) + b

    # Pack up data into dict
    ds = {
        'Function': 'y = tanh(x) + b',
        'x_gt': x_gt,
        'y_gt': y_gt,
        'x': x,
        'y': y,
    }

    return ds


def main():
    # Generate data
    N_OBS = 1000
    ds = generate_data(num_obs=N_OBS)
    plot_training_data(ds)

    # Split data into training and test sets
    train_data, test_data = None, None

    # Initialize network and run foward pass to make sure everything works
    num_layers = 3
    nn_mdl = init_network(num_layers=num_layers, num_inputs=1, num_units=[10, 3, 1], act_fn=['tanh', 'tanh', 'linear'])
    nn_mdl, y, states = forward_pass(mdl=nn_mdl, x=ds['x'])
    err_init = cost_fun(ds['y'], y)
    print('Initial error: {}'.format(err_init))

    # Plot predictions of initialized model
    ds_init = ds.copy()  # Need to do this in order to make a copy of ds (otherwise it is a reference)
    ds_init['y'] = y
    plot_training_data(ds_init, 'InitialPrediction')

    print(nn_mdl)
    nn_mdl, loss = train_network(nn_mdl, x=ds['x'], y=ds['y'])
    nn_mdl, y, states = forward_pass(mdl=nn_mdl, x=ds['x'])
    ds_init['y'] = y
    plot_training_data(ds_init, 'TrainedPrediction')

    return None


# Run main function
main()

# TODO - add different activation functions (relu, leakly relu) and re-run
# TODO - add some sort of display to indication epoch progression
# TODO - add plotting of cost/loss function