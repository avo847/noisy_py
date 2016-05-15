import numpy as np
import matplotlib.pyplot as plt
import gen_data as gen
import utility as util

def f(x):
  return np.sin(x)
  
xvals = np.arange(-2*np.pi, 2*np.pi, 0.1)
yvals = gen.noisy_1d(f, xvals, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
util.set_axis_lims(ax, np.array([xvals,yvals]))
ax.plot(xvals,yvals, 'o')
ax.plot(xvals,f(xvals))
plt.grid('on')
plt.show()

