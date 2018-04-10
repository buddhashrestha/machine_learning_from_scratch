# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:40:12 2018

@author: Huaming
"""


from sklearn.datasets import load_iris
from datetime import datetime

import time
start = time.time()
data = load_iris()

from sklearn.model_selection import train_test_split


import numpy as np
import operator
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], random_state=0)


def findNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(
            testInstance, trainingSet[x][0], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def euclideanDistance(instance1, instance2, length):
    distances = (instance1-instance2)**2
    distances = distances.sum(axis=-1)
    distances = np.sqrt(distances)
    return distances

def classifyLabel(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def diff(start, done):
    diff = done - start
    return diff.total_seconds()*1000


class KNN(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """
        X = X_train
        y = y_train
                z = merged value
        """
        self.X_train = X
        self.y_train = y
        self.train = zip(self.X_train, self.y_train)

    def predict(self, X_test, k=1):
        """
        It takes X_test as input, and return an array of integers, which are the
        class labels of the data corresponding to each row in X_test.
        Hence, y_project is an array of lables voted by their corresponding
        k nearest neighbors
        """
        y_predict = []
        for x in range(len(X_test)):
            neighbors = findNeighbors(self.train, X_test[x], k)
            label = classifyLabel(neighbors)
            y_predict.append(label)

        return y_predict

    def report(self, X_test, y_test, k=1):
        """
        return the accurancy of the test data.
        """
        my_predictions = self.predict(X_test, k)

        accuracy = getAccuracy(y_test, my_predictions)
        return accuracy

    
knn = KNN()
iterator = 1

def k_validate(X_test, y_test):
    """
    plot the accuracy against k from 1 to a certain number so that one could pick the best k
    """
    
    knn.train(X_train, y_train)
    all_predictions = []
    best_accuracy = 0
    best_k = 1
    print(X_train.shape[0])
    for i in range(1, X_train.shape[0]):
        iterator = i
        #%timeit knn.predict(X_test,iterator)
        accuracy = knn.report(X_test, y_test, i)
        if(best_accuracy <= accuracy):
            best_accuracy = accuracy
            best_k = i
        all_predictions.append(accuracy)
    #plotting the obtained accuracy for each correspoinding value of k
    print("Best accuracy : ", best_accuracy)
    print("Best k might be : ", best_k)
    # plt.figure(figsize=(50, 50))
    # plt.xlabel("Varied k")
    # plt.ylabel("Accuracy")
    # plt.plot(all_predictions)
    # plt.show()


k_validate(X_test, y_test)