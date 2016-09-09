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
ax.plot(xvals,yvals, 'o', label="sampled points")
ax.plot(xvals,f(xvals), label='original curve')


# Determine coefficients for a polynomial fit
#w = lrg.poly_fit_reg(xvals, yvals, 12, 0)
w = lrg.linear_basis_reg(xvals, yvals, np.power, 1, 12, 1, 0)
print 'coeffs: ', w 

fit = np.array([w[0][0]]*len(xvals))
for i in range(1, len(w)):
  fit += w[i][0] * xvals**i
  
# compute sum squared error
print 'mean square error: ', metrics.mean_sq_err(fit, yvals)
print "rms", metrics.root_mean_sq_err(fit,yvals)

# add polynomial fit to plot
ax.plot(xvals, fit, 'r-', label="polynomial fit")  
plt.grid('on')
plt.legend()
plt.show()