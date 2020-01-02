# Plotting module for neural analysis package

# Plot speed profile
def speed_profile(df):
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


# Plot reach trajectories
def reach_trajectories(df):
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
