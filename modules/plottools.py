# Plot tools module.


def create_subplot(
        n_row, n_col,
        x_marg = [100, 100],  # x-margin (pixels)
        y_marg = [100, 100],  # y-margin (pixels)
        ax_sp = 100,  # distance between subplots
        ax_w = 300.0,  # plot width (pixels)
        ax_h = 300.0,  # plot height (pixels)
        ):

    import numpy as np
    import matplotlib.pyplot as plt

    # Define plot shape, create figure
    fig_w = x_marg[0] + ax_w*n_col + (n_col - 1)*ax_sp + x_marg[1]
    fig_h = y_marg[0] + ax_h*n_row + (n_row - 1)*ax_sp + y_marg[1]
    dpi = 100
    fig_size = np.array([fig_w, fig_h])/dpi
    fh = plt.figure(figsize=fig_size, dpi=dpi)

    # Iterate over subplots and create axes
    ax_hndl = []  # Axes handles
    plot_ctr = 0
    for row in range(n_row):
        for col in range(n_col):
            # Define axis position. Axes are ordered left-to-right, top-to-bottom,
            # consistent with standard "subplot" style ordering.
            x_ax = x_marg[0]/fig_w + col*(ax_w + ax_sp)/fig_w
            y_ax = y_marg[0]/fig_h + (n_row - row - 1)*(ax_h + ax_sp)/fig_h
            ax_hndl.append(fh.add_axes([x_ax, y_ax, ax_w/fig_w, ax_h/fig_h]))

    return ax_hndl
