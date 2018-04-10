# -*- coding: utf-8 -*-
"""
Created on Tuesday March  20 17:30:23 2018

@author: Buddha Shrestha

This program is specifically designed to solve 2 class problems.
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA



def getMean(x,y,label):
    zipped = zip(x,y,label)
    classes = set(label)
    df = pd.DataFrame(list(zipped))

    mean = df.groupby(2).mean()
    mean = mean.iloc[:].values
    return mean



def driver():
    x = np.linspace(0,1,200)

    y = np.zeros_like(x,dtype = np.int32)

    x[0:100] = np.sin(4*np.pi*x)[0:100]

    x[100:200] = np.cos(4*np.pi*x)[100:200]

    y = 4*np.linspace(0,1,200)+1.5*np.random.randn(200)

    label= np.ones_like(x)

    label[0:100]=0

    plt.scatter(x,y,c=label)

    mean = getMean(x,y,label)

    mean1 = mean[0]
    mean2 = mean[1]

    #z1 = (x-mean1) and z2 = (x-mean2)
    z = np.array(zip(x,y)).T

    #compute the covariance matrix
    c = np.cov(z)
    #compute the inverse of covariance
    cinv = np.linalg.inv(c)

    #computing the coefficient terms
    r = mean2.T.dot(cinv).dot(mean2) - mean1.T.dot(cinv).dot(mean1)
    q = cinv.dot(mean2) - cinv.dot(mean1)
    p = mean2.T.dot(cinv) - mean1.T.dot(cinv)

    x = np.linspace(-1.0, 1.0, 200)
    y = np.linspace(-1.0, 5.0, 200)
    X, Y = np.meshgrid(x,y)

    #coefficients
    coeff = [p[i] + q[i] for i in range(len(p))]
    print("Equation of line is : %2.3f x + %2.3f y - %2.3f  = 0" % (coeff[0],coeff[1],r))
    #function
    F =  coeff[0] * X + coeff[1] * Y - r

    plt.contour(X,Y,F,[0])
    plt.show()

driver()