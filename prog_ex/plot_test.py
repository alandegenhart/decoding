# Plot test script.  This script is meant to be a test of various python and
# matplotlib plotting functionality.

# Add custom modules to the search path.  This is probably not the best way to
# do this, but is being used as a convenience for now.
import sys
sys.path.append('/Users/alandegenhart/Documents/GitHub/modules/')

# Import
import numpy as np
import matplotlib.pyplot as plt
import plottools as pt

# Matplotlib housekeeping -- set the renderer and the font type. Setting the
# renderer to 'PDF' (non-interactive) allows figures to be generated when
# running scripts from the command line. Setting the font types ensures that
# any text that is generated can be edited.
import matplotlib as mpl
mpl.use('PDF')
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Create some data to plot
x = np.linspace(-1.0, 1.0, 100)
y = x**2

# Sample data
x_samp = np.random.uniform(-1.0, 1.0, 1000)
e_samp = np.random.normal(0, 0.05, 1000)
y_samp = x_samp**2 + e_samp

# Create subplots
ax_hndl = pt.create_subplot(1, 2)

# Set first axis to be active
plot_no = 0
plt.sca(ax_hndl[plot_no])

# Plot -- scatter-style plot without a line
plt.plot(x_samp, y_samp, color=[0.75, 0.75, 0.75], marker='o', linestyle='None')
# Plot -- basic line with markers
plt.plot(x, y, color='k', linewidth=2)

# Format plot
x_lim = [-1, 1]
y_lim = [-0.25, 1.75]
x_tick = np.linspace(x_lim[0], x_lim[1], 4)
y_tick = np.linspace(y_lim[0], y_lim[1], 4)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.xticks(x_tick)
plt.yticks(y_tick)

# Turn off spines (borders)
ax_hndl[plot_no].spines['top'].set_visible(False)
ax_hndl[plot_no].spines['right'].set_visible(False)

plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.title('Title')

# Plot patch
# Set second axis to be active
plot_no = 1
plt.sca(ax_hndl[plot_no])

#  Define patch data.  This is a Nx2 matrix of vertices
x_patch = np.concatenate((x, x[::-1]))
y_patch = np.concatenate((y + .25, (y - .25)[::-1]))
v = np.vstack((x_patch, y_patch)).T
v = np.vstack((v, v[0, :]))  # Add first point to end

# Define path codes
c = [mpl.path.Path.LINETO for i in v]
c[0] = mpl.path.Path.MOVETO
c[-1] = mpl.path.Path.CLOSEPOLY

# Add patch
path = mpl.path.Path(v, c)
patch = mpl.patches.PathPatch(path,
                              facecolor=np.ones((3))*0.75,
                              alpha=0.5,
                              edgecolor=np.ones((3))*0.75)
# Note - setting any of the properties to 'None' will result in the default
# value in rcParams being used.
ax_hndl[plot_no].add_patch(patch)

# Plot underlying line
plt.plot(x, y, color='k', linewidth=2)

# Format plot
x_lim = [-1, 1]
y_lim = [-0.25, 1.75]
x_tick = np.linspace(x_lim[0], x_lim[1], 4)
y_tick = np.linspace(y_lim[0], y_lim[1], 4)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.xticks(x_tick)
plt.yticks(y_tick)

# Turn off spines (borders)
ax_hndl[plot_no].spines['top'].set_visible(False)
ax_hndl[plot_no].spines['right'].set_visible(False)

plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.title('Title')

# TODO: Add legend

# Set figure title (suptitle)
plt.suptitle('Suptitle')

# Save
fig_name = 'ReachTrajectories'
save_dir = '/Users/alandegenhart/Documents/GitHub/pythonExamples/results'
plt.savefig('results/{}.pdf'.format(fig_name))
