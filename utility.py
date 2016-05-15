import numpy as np
"""
Set axis limits based on given data points and plot data
"""
def set_axis_lims(ax, data):
  overhang = 0.5 # window overhang as percent of data width
  #num_pts = 1000. # number of points to use for plotting function
  xmin = np.min(data[0,:])
  xmax = np.max(data[0,:])
  ymin = np.min(data[1,:])
  ymax = np.max(data[1,:])
  
  delta_x = xmax - xmin
  delta_y = ymax - ymin
  
  xlim_low = xmin - overhang * delta_x
  xlim_high = xmax + overhang * delta_x
  ylim_low = ymin - overhang * delta_y
  ylim_high = ymax + overhang * delta_y
  ax.set_xlim(xlim_low, xlim_high)
  ax.set_ylim(ylim_low, ylim_high)
  
  #xvals = np.arange(xlim_low, xlim_high, delta_x/num_pts)
  #yvals = f(xvals)
  
  #ax.plot(xvals,yvals)