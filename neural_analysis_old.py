# Neural analysis

# Add modules directory to search path
import sys
sys.path.append('/Users/alandegenhart/Documents/GitHub/python/modules/')


# Load MGR data
def load_mgr_data():
    # Import libraries
    import pandas as pd

    # Define file path and load
    file_path = '/Volumes/Samsung_T5/Random Datasets/Ike_PMd_MemoryGuidedReach/Ike_MGR.hdf'
    df = pd.read_hdf(file_path)

    return df


def calc_movement_onset(df):
    import numpy as np
    import plottools as pt
    import matplotlib as mpl
    mpl.use('PDF')
    import matplotlib.pyplot as plt
    import copy

    # Define speed threshold used for movement onset
    s_thresh = 0.2

    # Create figure
    ax_hndl = pt.create_subplot(1, 1)

    # Iterate over trials and calculate speed
    n_trials = df.shape[0]
    s_all = []
    onset_idx = []
    for i in range(n_trials):
        # Get velocity, time, calculate speed
        v = df['vel'][i][0:2, :]
        t = df['time'][i]
        s = np.sqrt(np.sum(v**2, 0))

        # First get timing events -- trajectory onset/offset
        traj_onset = df['trajectoryOnset'][i]
        traj_offset = df['trajectoryOffset'][i]

        # Get time of max speed.  Find this between trajectory onset and offset
        s_temp = copy.copy(s)  # Create a copy of s
        s_temp[t < traj_onset] = 0  # zero out all speeds pre-trajectory onset (movement onset can't occur here)
        s_temp[t > traj_offset] = 0  # also zero-out all speeds post-trajectory offset
        max_ind = np.argmax(s_temp)  # Index corresponding to the max speed
        s_max = s[max_ind]
        t_max = t[max_ind]

        # Define masks used to find movement onset
        s_mask = s < (s_max * s_thresh)  # All speeds less than the threshold
        t_onset_mask = t > traj_onset
        t_max_mask = t < t_max
        mask = s_mask & t_onset_mask & t_max_mask  # All time points after trajectory onset but before max speed
        valid_idx = np.nonzero(mask)  # Get list of valid indices (speeds less than the threshold)
        onset_idx.append(valid_idx[0][-1] + 1)  # Movement onset will be the first time point after the last non-zero value

        # Add speed to list
        s_all.append(s)

    # Add movement onset and speed to dataframe
    df['speed'] = s_all
    df['onset_idx'] = onset_idx

    return None


def bin_trial_data(
        df,
        bin_width = 20,  # Bin width (ms)
        bin_start = 0,  # Start of bin times (ms)
        valid_sort = range(1, 31),  # Valid ranges of sort codes to use
        kin_offset = 0  # Temporal offset to apply to kinematic data
        ):
    """
    Bin trial data.

    This function bins neural and kinematic data.

    """
    # To access a single element in the dataframe df, use 'df.iloc[idx]'
    # Iterate over trials in df, determine the list of valid channel/sort

    # Iterate over trials
    for trial in df.iterrows():
        # Iterate over channel/sort combinations.  Note that the 'iterrows()'
        # function is not the most efficient way to parse a dataframe.  The
        # 'apply()' method can be used to apply a specific function to each
        # element.
        for ch, srt, st in zip(
            trial.spike_channel,
            trial.spike_sort,
            trial.spike_times
            ):

            # Check to see if ch and srt are in the valid list.  If so, bin the
            # spike times.
            X = []

        # Bin hand position data as well
        Y = []

    # Outputs
    bin_data = []  # Binned neural spike counts (dict, one element per trial)

    return X, Y


def bin_data(x, t, bin_width, bin_start):
    """
    Bin data.

    Bin the provided data into evenly-distributed time bins.

    """
    # Define time grid
    t_bins = []
    x_bins = []

    # Iterate through time bins

    return x_bins, t_bins


# Plotting functions

# Plot reach trajectories
def plot_reach_trajectories(df):
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # Get unique targets
    tc = df['targetCode'].values
    n_trials = len(tc)
    tc_unique = np.unique(tc)

    # Setup figure
    dpi = 100
    ax_size = [400, 300]  # axis size (pixels)
    fig_size = [600, 600]  # figure size (pixels)
    fig_size = tuple(np.array(fig_size)/dpi)
    fh = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fh.add_axes([0.15, 0.15, 0.7, 0.7])

    # Get plotting colors
    cmap = matplotlib.cm.get_cmap('hsv')
    n_targ = len(tc_unique)
    targ_col = cmap(np.linspace(0, 1, n_targ+1))

    # Iterate over unique target codes and plot targets
    patches = []
    for t in tc_unique:
        # Get target
        tc_mask = np.in1d(tc, t)
        pos = df['targetPosition'].values[tc_mask][0]
        target_radius = df['targetRadius'].values[tc_mask][0]

        # Plot target
        targ_col = cmap(t/(n_targ+1))
        circle = matplotlib.patches.Rectangle(pos - target_radius, target_radius*2, target_radius*2,
            facecolor=targ_col,
            linewidth=0.5,
            edgecolor=[0, 0, 0],
            alpha=0.5)
        ax.add_patch(circle)

    # Iterate over all trials and plot trajectories
    for t in range(n_trials):
        pos = df.pos[t]
        tc = df.targetCode[t]
        targ_col = cmap(tc / (n_targ + 1))
        plt.plot(pos[0], pos[1], color=targ_col, linewidth=0.5)
        plt.plot(pos[0, -1], pos[1, -1], 'ko', markersize=2)

    # Format plot and save
    ax_lim = (-200, 200)
    ticks = [-100, 0, 100]
    ax.set_xlim(ax_lim)
    ax.set_ylim(ax_lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.suptitle('Memory-guided reach trajectories')

    fig_name = 'ReachTrajectories'
    save_dir = '/Users/alandegenhart/Documents/GitHub/python/results'
    plt.savefig('results/{}.pdf'.format(fig_name))
    plt.close(fh)

    return None


# Plot speed profile
def plot_speed_profile(df):
    import numpy as np
    import plottools as pt
    import matplotlib as mpl
    mpl.use('PDF')
    import matplotlib.pyplot as plt
    import copy

    # Create figure
    ax_hndl = pt.create_subplot(1, 1)

    # Iterate over trials and calculate speed
    n_trials = df.shape[0]
    s_all = []
    for i in range(n_trials):
        # Get time, speed
        t = df['time'][i]
        s = df['speed'][i]

        # Align time so t=0 occurs at movement onset
        onset_idx = df['onset_idx'][i]
        t = t - t[onset_idx]

        # Plot trajectory
        plt.plot(t, s, 'k')
        plt.plot(t[onset_idx], s[onset_idx], 'ko')

    # Format figure
    plt.xlim([-500, 1000])
    plt.xlabel('Time (ms)')
    plt.ylabel('Speed')
    plt.suptitle('Reach speed')

    # Save figure
    fig_name = 'SpeedProfile'
    plt.savefig('results/{}.pdf'.format(fig_name))

    return None


# Main function
def main():
    # Load data
    df = load_mgr_data()
    print(df.head())

    # Print reach trajectories
    #plot_reach_trajectories(df)

    # Calc speed profile
    calc_speed_profile(df)

    return None


# Run main function
if __name__ == '__main__':
    main()
