"""
Neural analysis test script

This is a development script used to test out implementing certain
functionality.  Once functions are appropriately debugged here, they will be
moved into more permanent functions/modules.
"""

# Setup -- load modules

# Add custom modules to the search path.  This is probably not the best way to
# do this, but is being used as a convenience for now.
import sys
sys.path.append('/Users/alandegenhart/Documents/GitHub/modules/')

# Import
import numpy as np
import matplotlib.pyplot as plt
import plottools as pt
import neural_analysis.proc as naproc
import neural_analysis.plot as naplot

# Import data
df = naproc.load_mgr_data()

#-------------------------------------------------------------------------------
# Behavioral characterizations -- reach trajectories, speed profile

# Calcualte movement onset
naproc.calc_movement_onset(df)

# Plot reach trajectories and speed profile
naplot.plot_reach_trajectories(df)
naplot.speed_profile(df)
