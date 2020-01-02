"""
Neural analysis test script

This is a development script used to test out implementing certain functionality.  Once functions are appropriately
debugged here, they will be moved into more permanent functions/modules.
"""

# Add custom modules to the search path.  This is probably not the best way to
# do this, but is being used as a convenience for now.
import sys
sys.path.append('/Users/alandegenhart/Documents/GitHub/modules/')

# Import
import numpy as np
import matplotlib.pyplot as plt
import plottools as pt
import neural_analysis as na

# Plot reach trajectories
na.plot_reach_trajectories(df)