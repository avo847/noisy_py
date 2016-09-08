import numpy as np
import matplotlib.pyplot as plt
import gen_data as gen
import utility as util
import linear_reg as lrg
import metrics as metrics
import cross_validation as cv

a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([11,12,13,14,15,16,17,18,19,20])
c = np.vstack((a,b))

# test partitioning function
"""
for start in np.arange(0,12):
  [x1,x2] = cv.partition(a,0.5,start)
  print "k = ", start
  print x1, x2
  print " "
  
"""

# test Gaussian noise generator  
x = np.arange(-2*np.pi, 2*np.pi, 0.1)
y_true = np.sin(x)

for sigma in np.arange(0.1, 0.5, 0.1):
  y_data = gen.noisy_1d(np.sin, x, sigma)
  print "variance: ", sigma**2, ", mean square error: ", lrg.mean_sq_err(y_true, y_data)


# test randomize function
"""
x_data = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15],[16,17,18,19,20]])
y_data = np.array([10,20,30,40,50])
#[x_new,y_new] = cv.randomize(x_data,y_data,1)

"""


# test plotting randomized data points
"""
x = np.arange(-2*np.pi, 2*np.pi, 0.1)
y=np.sin(x)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(x,y, 'o' ,label="original data")

[x_rand,y_rand] = cv.randomize_data(x,y,0)

ax2 = fig.add_subplot(212)
ax2.plot(x_rand,y_rand, 'ro', label="randomized data")

plt.show()
"""