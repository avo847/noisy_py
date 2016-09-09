import numpy as np
import scipy.linalg as linalg
import utility as util
"""
Maximum likelihood linear basis function 
fit to data using polynomials up to and including 
degree max_deg
"""
def poly_fit_ml(xvals, yvals, max_deg):
  xvals = np.transpose(xvals[np.newaxis]) # make xvals into 2d array and transpose
  yvals = np.transpose(yvals[np.newaxis])
  
  # Construct "design matrix"
  # np.newaxis turns 1d row vector into n x 1 column vector
  phi = np.transpose( np.array([1] * len(xvals))[np.newaxis] )
  for i in np.arange(1,max_deg+1):
    phi = np.hstack( [phi, np.power(xvals, i)] )
  
  # need to solve phi.T * phi * w = phi.T * yvals
  coeffs = np.linalg.solve( np.dot(phi.T, phi), np.dot(phi.T,yvals))
  return coeffs
  
  
"""
Regularlized least square fit with polynomials
"""
def poly_fit_reg(xvals, yvals, max_deg, lam=0):
  xvals = np.transpose(xvals[np.newaxis]) # make xvals into 2d array and transpose
  yvals = np.transpose(yvals[np.newaxis])
  
  # Construct "design matrix"
  phi = np.transpose( np.array([1] * len(xvals))[np.newaxis] )
  for i in np.arange(1,max_deg+1):
    phi = np.hstack( [phi, np.power(xvals, i)] )
  
  # need to solve (phi.T * phi + lamda* I) * w = phi.T * yvals
  #coeffs = util.solve( np.dot(phi.T, phi) + lam * np.identity(max_deg+1), np.dot(phi.T,yvals))
  L = lam * np.identity(max_deg+1)
  L[0,0] = 0 # dont penalize parameter corresponding to mean
  coeffs = np.linalg.solve( (np.dot(phi.T, phi) + L), np.dot(phi.T,yvals))
  return coeffs

"""
Least squares fit using a linear combination of functions of the same form.
xvals: vector of independent variable values
yvals: voctor of correspoding values of response variable
function(xvals, index): a 1-parameter family of functions. The indexing variable
  must be uniformly spaced
min: minimum value to use for index
max: maximum value to use for index
step: step size between indexing values
"""
def linear_basis_reg(xvals, yvals,func, start, stop, step=1, lam=0):
  xvals = np.transpose(xvals[np.newaxis]) # make xvals into 2d array and transpose
  yvals = np.transpose(yvals[np.newaxis])
  
  phi = np.transpose( np.array([1] * len(xvals))[np.newaxis] ) # first column is all 1s
  for i in np.arange(start,stop+step, step):
    phi = np.hstack( [phi, func(xvals, i)] )
    
  print "shape(phi): ", np.shape(phi)
    
  # need to solve (phi.T * phi + lamda* I) * w = phi.T * yvals
  n =int( float( stop-start) / step + 2)
  L = lam * np.identity( n)
  print "n: ", n 
  print "size(yvals)", np.shape(yvals)
  print "shape(L): ", np.shape(L)
  L[0,0] = 0 # dont penalize parameter corresponding to mean
  coeffs = np.linalg.solve( (np.dot(phi.T, phi) + L), np.dot(phi.T,yvals))
  return coeffs
  



