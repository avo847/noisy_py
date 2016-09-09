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
yvals = gen.noisy_1d(f, xvals, 0.2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xvals,yvals, 'o', label="sampled points")
ax.plot(xvals,f(xvals), label='original curve')


# One possible family of basis functions, indexed by center mu
def g(x, mu, sigma):
  return np.exp(-(x-mu)**2/ sigma**2 )

sigma =0.5

# Determine coefficients for a polynomial fit
#w = lrg.linear_basis_reg(xvals, yvals, 0, np.power, 1,12,1)
w = lrg.linear_basis_reg(xvals, yvals, 0, g, -6.,6., 2., sigma)
print 'coeffs: ', w 

# Call function that performs fitting using obtained coefficients
#fit = lrg.fit_linear_model(xvals, w, np.power, 1,12,1)
fit = lrg.fit_linear_model(xvals, w, g, -6., 6., 2.,sigma)

# compute sum squared error
print 'mean square error: ', metrics.mean_sq_err(fit, yvals)
print "rms", metrics.root_mean_sq_err(fit,yvals)

# add polynomial fit to plot
ax.plot(xvals, fit, 'r-', label="fitted model")  
plt.grid('on')
plt.legend()
plt.show()