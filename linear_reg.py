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
