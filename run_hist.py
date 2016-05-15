import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import gen_data as gen
import utility as util

def f(x):
  return np.sin(x)

numpts = 1000
mu, sigma = 0., 10.
x = mu + sigma * np.random.randn(numpts)
data = mu + sigma * gen.gaussian_noise_1d(numpts)
print type(data)
# the histogram of the data
n, bins, patches = plt.hist(data, 50, normed=1, facecolor='green', alpha=0.75)
#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='blue', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.grid(True)

plt.show()