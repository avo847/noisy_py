import numpy as np
import linear_reg as lrg
import matplotlib.pyplot as plt
"""
Partition a numpy array into two such arrays, 
either by proportion or by number.
If a proportion is used the proportion of total s rounded 
down to closest integer below that fraction of total
number of elements.

"""
def partition(x, p, start = 0):
  start = int(start)

	
  numpts = len(x)
  if start > numpts:
    start = np.mod(start,numpts)
	
  if (0<=p < 1):
    many = p * numpts
  else:
    many= p
  
  if many > numpts:
    print "warning: asking for subarray larger than given array,"
    print "using modulus instead"
    many = np.mod(many,numpts)
	
  many = int(many)
  if start+many > numpts:
    x1= np.hstack( (x[start:numpts], x[0:np.mod(start+many, numpts)]))
    x2= np.delete( x, np.arange(start,numpts))
    x2= np.delete(x2, np.arange(0, np.mod(start+many, numpts)))
  else:
    x1 = x[start:start+many] # value indexed by end not included
    x2 = np.delete( x, np.arange(start, start+many))
  
  return [x1,x2]


"""
k-fold cross validation:
split data set into k instance of training and test data.
Use training set to fit the model and test set to compare 
performance on a particular metric.

metric: used to determine efficacy of model
k: number of subsets to split data into
rand: (True/False) randomize points in data set if true
model: function which computes fit from data sets
x_vals: input deta
y_vals: target/response data
*args: additional inputs to fitting function

Note: 
"""
def kcv(metric, k, rand, model, x_vals, y_vals, *args):
  numpts = len(y_vals)
  set_size = np.ceil(numpts / float(k) )# number of elements in test set
                                                      # use ceil because test set may be small
  np.vstack((x_vals,y_vals)) # throw exception here if different sizes
  
  # sort data set for plotting
  p = x_vals.argsort() # determine required permuation to order x data
  x_sort = x_vals[p]
  y_sort = y_vals[p]
  
  # randomize data set
  if rand:
    [x_vals,y_vals] = randomize_data(x_vals,y_vals,0)
  
  # spli data into training set and test set
  for i in np.arange(0,k):
    [x_test, x_train] = partition(x_vals, set_size, i*set_size)
    [y_test, y_train] = partition(y_vals, set_size, i*set_size)
    
    # Determine coefficients for a polynomial fit
    w = model(x_train, y_train, *args)
    
    # Compute predicted outputs on test inputs, based on given polynomial model
    fit = np.array([w[0][0]]*len(x_test))
    for j in range(1, len(w)):
      fit += w[j][0] * x_test**j
    
    # Compute predicted values on all inputs, for plotting purposes  
    full_fit = np.array([w[0][0]]*len(x_sort))
    for j in range(1, len(w)):
      full_fit += w[j][0] * x_sort**j
      
      
    # print mean square error on test data
    print "iter: ", i
    print "mean square error: ", metric(fit, y_test)
    
    # plot training data, test data and fitted model on each run
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #util.set_axis_lims(ax, np.array([xvals,yvals]))
    ax.plot(x_train, y_train, 'o', label="training data")
    ax.plot(x_sort,np.sin(x_sort), label='generated from')
    
    ax.plot(x_test, y_test, 'ro', label="test data")
    # add polynomial fit to plot
    ax.plot(x_sort, full_fit, 'r-', label="polynomial fit")  
    
    plt.grid('on')
    plt.legend()
    plt.show()
  
  return 1
  
  
""" 
Randomize covariates and response variable together.
User has option to input seed 

y_data is assumed to be 1d numpy array
x_data can be 2d array with each column corresponding to one data point

"""
def randomize_data(x_data, y_data, seed=0):
  numpts = len(y_data)
  x_out = np.zeros(x_data.shape)
  y_out = np.zeros(y_data.shape)
  
  np.random.seed(seed)  
  index_in = np.random.permutation(numpts)
  #print "cv. randomize: permuted indicies ",  index_in
  
  # perrmute columns of x_data and y_data according to the above permutation
  index_out = np.arange(0,numpts)
  if len(x_data.shape) == 1:
    x_out[index_out] = x_data[index_in]
  else:
    x_out[:,index_out] = x_data[:, index_in]
  y_out[index_out] = y_data[index_in]
  
  return [x_out,y_out]
