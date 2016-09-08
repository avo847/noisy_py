import numpy as np

def root_mean_sq_err(x,y):
  x = x.astype(float)
  numpts = len(x)
  val = np.square(x-y).sum() / numpts
  return np.sqrt(val)
  
def euclidean(x,y):
  x = x.astype(float)
  val = np.square(x-y).sum()
  return np.sqrt(val)
  
  
def sum_sq_err(fit, targets):
  err = fit - targets
  return np.dot(err, err)
  
def mean_sq_err(fit, targets):
  return sum_sq_err(fit,targets) / targets.size