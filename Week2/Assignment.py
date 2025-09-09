import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
%matplotlib widget
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from public_tests import * 

from autils import *
from lab_utils_softmax import plt_softmax
np.set_printoptions(precision=2)

# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ### 
    exp_z = np.exp(z - np.max(z))
    a = exp_z / np.sum(exp_z)
    ### END CODE HERE ### 
    return a
    
    
        
    def my_softmax(z):  
    N = len(z)
    a =  np.zeros(N)                  # initialize a to zeros 
    ez_sum = 0               # initialize sum to zero
    for k in range(N):      # loop over number of outputs             
        ez_sum += np.exp(z[k] )          # sum exp(z[k]) to build the shared denominator      
    for j in range(N):      # loop over number of outputs again                
        a[j] = np.exp(z[j] ) / ez_sum              # divide each the exp of each output by the denominator   
    return(a)

        
    
    
    ### END CODE HERE ### 
    return a

# load dataset
X, y = load_data()

# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.layers.Input(shape=(400,)),
        Dense(25, activation='relu',name="L1"),
        Dense(15, activation='relu',name="L2"),
        Dense(10, activation='linear',name="L3")
        ### END CODE HERE ### 
    ], name = "my_model" 
)