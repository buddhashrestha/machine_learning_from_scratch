#!/usr/bin/env python

import svmpy
import kernel
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import argh

import cloudpickle as pickle
mnist23 = pickle.load( open( "../datasets/mnist23.data", "rb" ) )
mnist23.data = mnist23.data[:700]
mnist23.target = mnist23.target[:700]
training_samples = 500
# Reshape the training and test examples 
train_x_flatten = mnist23.data[:training_samples]  #train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
train_y = np.array([mnist23.target[:training_samples]]).T
test_x_flatten = mnist23.data[training_samples:] 
test_y = np.array([mnist23.target[training_samples:]]).T  #test_x_orig.reshape(test_x_orig.shape[0], -1).T
test_y = test_y - 2
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

def example(num_samples=10, num_features=784, grid_size=20, filename="svm.pdf"):
    samples = np.matrix(np.random.normal(size=num_samples * num_features)
                        .reshape(num_samples, num_features))
    samples = train_x
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0
    labels = train_y - 2
    print("sample variance :", np.var(samples))
    
    trainer = svmpy.SVMTrainer(kernel.Kernel()._polykernel(120,1), 0.1)
    predictor = trainer.train(samples, labels)
    predictAll(test_x,test_y ,predictor)
    # plot(predictor, samples, labels, grid_size, filename)


def plot(predictor, X, y, grid_size, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(filename)

def predictAll(X, y, predictor):
    m = X.shape[0]
    print("Total test set ::",m)
    p = np.zeros((1, m),dtype=int)
    for i in range(0, m):
        p[0,i] = predictor.predict(X[i])
    print("prediction :",p)
    print("Accuracy: %s" % str(np.sum(p == y)/float(m)))

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    argh.dispatch_command(example)