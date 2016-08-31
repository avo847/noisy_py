import numpy as np
import matplotlib.pyplot as plt
import gen_data as gen
import utility as util
import linear_reg as lrg
import metrics as metrics
import cross_validation as cv

def f(x):
  return np.sin(x)
  
xvals = np.arange(-2*np.pi, 2*np.pi, 0.1)
yvals = gen.noisy_1d(f, xvals, 1)


fig = plt.figure()
ax = fig.add_subplot(111)
#util.set_axis_lims(ax, np.array([xvals,yvals]))
ax.plot(xvals,yvals, 'o', label="sampled points")
ax.plot(xvals,f(xvals), label='generated from')

plt.grid('on')
plt.legend()
plt.show()


cv.kcv(metrics.euclidean, 5, lrg.poly_fit_reg, xvals, yvals, 8, 0)