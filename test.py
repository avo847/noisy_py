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

"""
for start in np.arange(0,12):
  [x1,x2] = cv.partition(a,0.5,start)
  print "k = ", start
  print x1, x2
  print " "
  
"""
#cv.kcv(metrics.euclidean, 4, lrg.poly_fit_reg, a,b)

x_data = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15],[16,17,18,19,20]])
y_data = np.array([10,20,30,40,50])

print x_data
print y_data
print 

[x_new,y_new] = cv.randomize(x_data,y_data,1)

print 
print x_new
print y_new