import numpy as np

"""
Sample points from function f 
on xvals (a numpy array) with normally 
distributed noise of variance sigma^2
"""

def noisy_1d(f, xvals, sigma):
  yvals = f(xvals)
  numpts=len(xvals)
  noise=gaussian_noise_1d(numpts)
  return yvals + sigma*np.array(noise)

"""
Use Box-Muller method for generating samples
from a Gaussian distribution of mean 0 and 
variance 1
"""
def gaussian_noise_1d(numpts):
  np.random.seed(7)
  i = 0
  data=[]
  while(i <= numpts):
    # First generate points in unit circle
    while(True):
      [z1,z2]= 2*np.random.rand(1,2)[0]-np.array([1,1])
      r_sq = z1**2 + z2**2
      if r_sq < 1:
        break
    # Now transform them to normally distributed values
    y1 = z1 * np.sqrt(-2*np.log(r_sq)/r_sq)
    y2 = z2 * np.sqrt(-2*np.log(r_sq)/r_sq)
    
    data.append(y1)
    data.append(y2)
    i+=2
    
  return np.array(data[0:numpts])
  