import time
import numpy as np
import matplotlib.pyplot as plt

from dnn_app_utils import *
start = time.clock()

import cloudpickle as pickle
mnist23 = pickle.load( open( "./datasets/mnist23.data", "rb" ) )
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
index = 10

print ("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")
# mnist23.data = mnist23.data[:5000]
# mnist23.target = mnist23.target[:5000]
training_samples = 10000

# Explore your dataset 
# m_train = train_x_orig.shape[0]
# num_px = train_x_orig.shape[1]
# m_test = test_x_orig.shape[0]

# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
train_x_flatten = mnist23.data[:training_samples]  #train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
train_y = np.array([mnist23.target[:training_samples]])
test_x_flatten = mnist23.data[training_samples:]
test_y = np.array([mnist23.target[training_samples:]]) #test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.

mean = train_x_flatten.mean(axis=0)

train_x =  (train_x_flatten - mean) / 255 
test_x =  (test_x_flatten - mean) / 255 
train_y = train_y - 2
test_y = test_y - 2

print("trainx :",train_x.shape)
pca = PCA(n_components=250)
pca.fit(train_x)
train_x = pca.transform(train_x)
print("trainx :",train_x.shape)
sys.exit()
test_x = pca.transform(test_x)
train_x = train_x.T
test_x = test_x.T
### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
### CONSTANTS ###
layers_dims = [train_x.shape[0], 20, 7, 3, 1] #  5-layer model
# layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
# GRADED FUNCTION: n_layer_model


def L_layer_model(X, Y, layers_dims, learning_rate=0.085, num_iterations=5000, print_cost=False): #lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    threshold = 0.0001
    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    return parameters


# parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

parameters = L_layer_model(train_x,train_y , layers_dims, num_iterations=3000, print_cost=True)

pred_train = predict(train_x, train_y, parameters)

pred_train = predict(test_x, test_y, parameters)
print("Everything good, time taken :", time.clock() - start)

