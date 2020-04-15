# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import plottools as pt


def gen_training_ex(n_seq, mode='radial'):
    """Generate training examples.
    
    Each training example consists of a pair of 2D time series. The input is
    a persistent signal that indicates the current desired cursor position.
    The output is the desired cursor position.
    
    The onset of the trajectories are randomized so that the network isn't
    simply able to learn the task timing w/o using the input signal.
    
    Usage:
        X, Y = gen_training_ex()
        
    Inputs:
        :n_seq -- number of sequences to generate
        :mode -- specify how sequences are generated
    """
    
    # Figure out what mode is being used. If the mode is 'radial', generate
    # trials that are diametrically-opposed (e.g., they go from one side of a
    # unit circle to the other)
    if mode == 'radial':
        r_start = np.linspace(0, 2*np.pi, n_seq + 1)[0:-1]
        r_end = r_start + np.pi
        
        # Convert radial positions to x and y coordinates
        x_start = np.cos(r_start)
        y_start = np.sin(r_start)
        x_end = np.cos(r_end)
        y_end = np.sin(r_end)
        
    # Determine onset. The mean onset time will be 1s, but the onset time for
    # individual trials will be drawn from a Gaussian distribution.
    t_onset = 1000
    t_onset_std = 100
    onset = np.random.normal(t_onset, t_onset_std, (n_seq,))
    
    # Determine sequence length
    t_step = 10  # Time step (ms)
    t_move = 500
    t_end = 2 * t_onset + t_move
    seq_len = int(t_end / t_step)
    onset = (onset / t_step).round().astype(int)  # Convert to onset index
    offset = (onset + (t_move / t_step)).round().astype(int)
    
    # Determine speed
    s_0 = 2/(t_move/t_step)
    
    # Iterate over sequences
    X = np.zeros((n_seq, seq_len, 2))
    Y = np.zeros((n_seq, seq_len, 2))
    for seq in range(n_seq):
        # Determine input. This will switch from the start to end position at
        # the onset time.
        x = np.full((seq_len, 2), 0.)
        idx = range(0, onset[seq])  # Range of indices for onset
        x[idx, 0] = x_start[seq]
        x[idx, 1] = y_start[seq]
        x[onset[seq]:, 0] = x_end[seq]
        x[onset[seq]:, 1] = y_end[seq]
        
        # Find unit vector between start and end targets and calculate speed
        # profile
        v = (np.array([x_end[seq], y_end[seq]]) - 
             np.array([x_start[seq], y_start[seq]]))
        d = np.linalg.norm(v)
        v = v / d  # Make unit vector
        s = speed_profile(d, s_0)
        
        # Convert speed profile into cursor positions
        v = np.tile(np.expand_dims(v, axis=0), (s.shape[0], 1))
        s = np.tile(np.expand_dims(s, axis=1), (1, 2))
        v = v * s  # Scaled velocity
        y_ramp = np.cumsum(v, axis=0)  + x[0, :]  # Integrate to get position
        
        # Create final cursor position profile
        y = x.copy()
        n_samp = y_ramp.shape[0]
        ind = range(n_samp) + onset[seq]
        y[ind, :] = y_ramp

        # Add to output matrices
        X[seq, :, :] = x
        Y[seq, :, :] = y

    # Put data into a dict
    onset = onset - 1  # Onset is the first sample before movement starts
    train_data = {'X': X, 'Y': Y, 'onset': onset, 'offset': offset}
    
    return train_data


def speed_profile(d_0, s_0, mode='constant'):
    """Generate speed profile.
    
    This function generates a speed profile with the profile specified by
    MODE.
    """
    n = 1000  # Number of time points.  For now just use something big.
    
    # Generate speed profile for specified mode
    if mode == 'constant':
        # Generate vector of constant speed and integrate to get distance
        s = np.full((n,), s_0)
        d = np.cumsum(s)
        
        # Truncate to valid distance
        mask = d <= d_0
        s = s[mask]
        d = d[mask]
        s = np.append(s, d_0 - d[-1])  # Add final point
        
    return s
        

def plot_training_ex(train_data, plot_title, axh=None):
    """Plot input/output data for simulated BCI trials."""
    
    # Get pointers to data (makes the following code easier to read)
    X = train_data['X']
    Y = train_data['Y']

    # Setup panels using 'gridspec'
    fh = plt.figure(figsize=[10, 10])
    gs = fh.add_gridspec(4, 4)
    axh = [None for _ in range(6)]
    axh[0] = fh.add_subplot(gs[0,0:2])  # Input - x
    axh[1] = fh.add_subplot(gs[0,2:])  # Output - x
    axh[2] = fh.add_subplot(gs[1,0:2]) # Input - y
    axh[3] = fh.add_subplot(gs[1,2:])  # Output - y
    axh[4] = fh.add_subplot(gs[2:,0:2])
    axh[5] = fh.add_subplot(gs[2:,2:])

    # Set axis labels
    axh[0].set_title('Input (target)')
    axh[1].set_title('Output (position)')
    axh[2].set_xlabel('Time')
    axh[2].set_ylabel('Input/Output')
    fh.suptitle(plot_title, fontsize=14, fontweight='bold')

    # Get colors for each example
    n_ex = X.shape[0]

    # Loop over input/output pairs and plot
    for i in range(n_ex):
        # Get color corresponding to the starting position
        col = get_color(X[i, 0, :])

        # Plot individual traces vs time
        axh[0].plot(X[i, :, 0], color=col)
        axh[2].plot(X[i, :, 1], color=col)
        axh[1].plot(Y[i, :, 0], color=col)
        axh[3].plot(Y[i, :, 1], color=col)

        # Plot 2D output
        if X[i, 0, 1] >= 0:
            ax = axh[4]
        else:
            ax = axh[5]

        # Get trajectory to plot
        traj = Y[i, train_data['onset'][i]:train_data['offset'][i], :].T
        plot_trajectory(traj, X[i, 0, :], ax)
    
    # Adjust axes so that the aspect ratio is equal for 2D plots
    ticks = [-1, 0, 1]
    for i in range(4, 6):
        axh[i].axis('equal')
        axh[i].set_xlim(-1.5, 1.5)
        axh[i].set_ylim(-1.5, 1.5)
        axh[i].set_xticks(ticks)
        axh[i].set_yticks(ticks)
        axh[i].set_xlabel('Output position (x)')
        axh[i].set_ylabel('Output position (y)')

    # Set title for 2D plots
    axh[4].set_title('Output (start position y >= 0)')
    axh[5].set_title('Output (start position y < 0)')

    # Format plots -- set axis limits for plots vs time
    ax_labels = [
        'Input (x)', 'Output (x)', 'Input (y)', 'Output (y)'
    ]
    for i in range(4):
        axh[i].set_xlim(0, X.shape[1])
        axh[i].set_ylim(-1.5, 1.5)
        axh[i].set_xlabel('Time')
        axh[i].set_ylabel(ax_labels[i])
        axh[i].set_yticks(ticks)

    # Remove right and top spines
    for i in range(6):
        axh[i].spines['top'].set_visible(False)
        axh[i].spines['right'].set_visible(False)

    # Adjust whitespace
    gs.tight_layout(fh, rect=[0, 0, 1, .95])

    return fh


def plot_trajectory(traj, pos, axh):
    """Plot the trajectory for a single simulated trial.
    
    Inputs:
        traj    Trajectory to plot [2 x time]
        pos     Starting position (typically the input/target)
        fh      Figure handle

    """
    # Get the appropriate color to plot
    col = get_color(pos)

    # Plot
    axh.plot(traj[0, :], traj[1, :], color=col)
    axh.plot(traj[0, 0], traj[1, 0], marker='.', color=col)
    axh.plot(traj[0, -1], traj[1, -1], marker='x', color=col)

    return None


def get_color(pos):
    """Generate colors for training examples.
    
    Use a HSV color scheme.  Define the hue and then convert to RGB.  Use a
    color encoding scheme based on the starting position of the trajectory.

    Inputs:
        pos   2-element vector with the starting position of the input
    
    """
    import colorsys

    # Convert the input position to an angle
    ang = np.arctan2(pos[1], pos[0])
    if ang < 0: ang = ang + 2*np.pi  # Now over the interval [0 2*pi]
    h = ang / (2 * np.pi)  # Now over [0 1] (hue)

    # Convert to RGB and return
    return colorsys.hsv_to_rgb(h, 1, 1) 


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
    In order to allow arbitrary-length sequences to be predicted, the batch
    shape should be [None, None, n_inputs].

    A note on regularization -- there are two different regularizers that can
    be applied.  The 'kernel' regularizer is applied to the weights that map
    the inputs to the recurrent units, and the 'recurrent kernel' are the
    weights of the recurrent units.
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
        tf.keras.layers.GaussianNoise(0.1),
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


def pca_analysis(model, data):
    """Run PCA analysis on model to visualize hidden layer activity.

    To get the values of the intermediate layer, a new model needs to be 
    created. This model takes the normal input from the RNN, and returns the
    output of the rnn layer.  The values of the hidden layer can then be found
    using the 'predict()' method.
    """
    
    # Unpack train data dict
    X = data['X']
    Y = data['Y']

    # --- Get hidden layer activations and run PCA ---

    # Create new model and predict input
    inputs = model.input
    outputs = [model.layers[0].output, model.layers[1].output]
    act_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    activations = act_model.predict(X)

    # Format activations into dimensions x observations
    # Input dimensions: (obs x time x units)
    n_inputs = X.shape[2]
    n_obs = X.shape[0]
    n_ts = X.shape[1]
    n_rnn = activations[0].shape[2]
    A = np.transpose(activations[0], (2, 0, 1))  # Now (units x obs x time)
    Y = np.transpose(activations[1], (2, 0, 1))
    A = A.reshape((n_rnn, -1), order='C')  # Now (units x (obs*time))
    Y = Y.reshape((n_inputs, -1), order='C')

    # Run PCA (note that this is actually PPCA)
    from sklearn.decomposition import PCA
    n_pcs = 20
    pca = PCA(n_components=n_pcs)
    pca.fit(A.T)

    # --- Figure 1: Variance explained ---

    # Setup figure
    fh, ax_h = pt.create_subplot(n_row=1, n_col=1)

    # Plot fraction of variance explained
    idx = 0
    dims = np.arange(pca.n_components_) + 1
    ax_h[idx].plot(dims, pca.explained_variance_ratio_, color='k', marker='.')
    ax_h[idx].set_xlabel('PCA dim.')
    ax_h[idx].set_ylabel('Fraction of variance explained')
    ax_h[idx].set_title('Fraction of variance explained')
    fig_name = 'FracVarExplained'
    fh.savefig('./results/BCISimulation_PCAResults_{}.pdf'.format(fig_name))

    # --- Figure 2: 2D projections of the PCA representation ---

    # Setup figure.  Define dimensions to plot
    plot_dim = [[0, 1], [0, 2], [1, 2], [2, 3]]
    n_row = 2
    n_col = len(plot_dim)
    fh, ax_h = pt.create_subplot(n_row=n_row, n_col=n_col)
    
    # Find indices of specific trials to plot.  Here we want to focus on a
    # pair of diametrically-opposed targets (0 and 180 degrees)
    start_pos = X[:, 0, :]
    start_ang = np.rad2deg(np.arctan2(start_pos[:, 1], start_pos[:, 0]))
    mask = start_ang < 0
    start_ang[mask] = start_ang[mask] + 360
    targ_ang = [0, 180]
    targ_idx = [np.argmin(np.abs(start_ang - ang)) for ang in targ_ang]

    # Iterate over trials
    Z = np.zeros((n_obs, n_ts, n_pcs))
    n_samp = np.zeros((n_obs), dtype=int)
    for i in range(n_obs):
        # Get data from current trial and find PCA representation
        A_trial = activations[0][i, :, :]
        Z_trial = pca.transform(A_trial)
        Z[i, :, :] = Z_trial

        # Limit to valid portion of the trajectory
        Z_trial = Z_trial[data['onset'][i]:data['offset'][i]]
        n_samp[i] = Z_trial.shape[0]

        # Iterate over dimensions and plot
        for ax, d in zip(ax_h[0:n_col], plot_dim):
            plot_trajectory(Z_trial[:, d].T, X[i, 0, :], ax)

        # If trial is to be highlighted, plot in a separate set of axes
        if i in targ_idx:
            for ax, d in zip(ax_h[n_col:], plot_dim):
                plot_trajectory(Z_trial[:, d].T, X[i, 0, :], ax)

    # Set axis labels
    for ax, d in zip(ax_h, plot_dim * 2):
        ax.set_xlabel('PC {}'.format(d[0] + 1))
        ax.set_ylabel('PC {}'.format(d[1] + 1))
        ax.set_title('Dim {} vs Dim {}'.format(d[0] + 1, d[1] + 1))

    # Save figure
    fig_name = '2DProj'
    fh.savefig('./results/BCISimulation_PCAResults_{}.pdf'.format(fig_name))

    # --- Figure 3: Linear mapping of PCA representation ---

    # Get data to fit linear model
    n_samp_all = np.sum(n_samp)
    Z_mat = np.zeros((n_samp_all, n_pcs))
    Y_mat = np.zeros((n_samp_all, n_inputs))
    ctr = 0
    for i in range(n_obs):
        onset_idx = np.arange(data['onset'][i], data['offset'][i])
        ctr_idx = np.arange(ctr, (ctr + len(onset_idx)))
        Z_mat[ctr_idx, :] = Z[i, onset_idx, :]
        Y_mat[ctr_idx, :] = data['Y'][i, onset_idx, :]
        ctr = ctr_idx[-1] + 1

    # Fit linear model -- do this independently for the X and Y dimensions
    from sklearn import linear_model
    reg_mdl = linear_model.LinearRegression(fit_intercept=False)
    reg_mdl.fit(Z_mat, Y_mat)
    r2 = reg_mdl.score(Z_mat, Y_mat)
    print('Linear fit: r2 = {}'.format(r2))
    W = reg_mdl.coef_
    
    # Plot predicted trajectories
    fh, ax_h = pt.create_subplot(n_row=1, n_col=1)
    for i in range(n_obs):
        # Predict cursor position from hidden unit activity
        Z_temp = Z[i, data['onset'][i]:data['offset'][i], :]
        y_pred = Z_temp @ W.T
        y_pred = reg_mdl.predict(Z_temp)
        plot_trajectory(y_pred.T, data['X'][i, 0, :], ax_h[0])

    # Format plot axes
    ax_h[0].set_title('Linear model - predicted trajectories')
    ax_h[0].set_xlabel('X position')
    ax_h[0].set_ylabel('Y position')

    # Save figure
    fig_name = 'LinearMapping'
    fh.savefig('./results/BCISimulation_PCAResults_{}.pdf'.format(fig_name))

    return None

