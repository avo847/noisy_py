import numpy as np
import matplotlib.pyplot as plt
import gen_data as gen
import utility as util
import linear_reg as lrg

def f(x):
  return np.sin(x)
  
xvals = np.arange(-2*np.pi, 2*np.pi, 0.1)
yvals = gen.noisy_1d(f, xvals, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
#util.set_axis_lims(ax, np.array([xvals,yvals]))
ax.plot(xvals,yvals, 'o', label="sampled points")
ax.plot(xvals,f(xvals), label='original curve')


# Determine coefficients for a polynomial fit
w = lrg.poly_fit_reg(xvals, yvals, 12, 0)
print 'coeffs: ', w 


fit = np.array([w[0][0]]*len(xvals))
for i in range(1, len(w)):
  fit += w[i][0] * xvals**i
  
# compute sum squared error
print 'mean square error: ', lrg.mean_sq_err(fit, yvals)

# add polynomial fit to plot
ax.plot(xvals, fit, 'r-', label="polynomial fit")  
plt.grid('on')
plt.legend()
plt.show()