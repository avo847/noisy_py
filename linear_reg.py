import numpy as np

"""
Maximum likelihood linear basis function 
fit to data using polynomials up to and including 
degree max_deg
"""
def poly_fit(xvals, yvals, max_deg, init_params):
  data = np.array([1] * len(xvals))
  for i in np.arange(0,max_deg+1):
    data = vstack( [data, np.power(xvals, i)] )
  
  # need to fix design matrix above