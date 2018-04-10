import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



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

    y = 4*np.linspace(0,1,200) + 1.5 * np.random.randn(200)

    label= np.ones_like(x)

    label[0:100]=0

    plt.scatter(x,y,c=label)


    mean = getMean(x,y,label)
    mean1 = mean[0]
    mean2 = mean[1]

    z1 = zip(x[0:100],y[0:100])
    z2 = zip(x[100:200],y[100:200])

    u1 = np.array(z1).T
    u2 = np.array(z2).T

    #compute covariance matrix
    c1 = np.cov(u1)
    c2 = np.cov(u2)

    #compute inverse of covariance matrix
    c1inv = np.linalg.inv(c1)
    c2inv = np.linalg.inv(c2)

    #compute the determinants of the matrix
    c1_det = np.linalg.det(c1)
    c2_det = np.linalg.det(c2)

    #computing the coefficient terms
    r = mean2.T.dot(c2inv).dot(mean2) - mean1.T.dot(c1inv).dot(mean1)
    q = 0.5 * np.log(c2_det/c1_det)
    A = mean2.T.dot(c2inv) - mean1.T.dot(c1inv)
    B = c2inv.dot(mean2) - c1inv.dot(mean1)
    C = c1inv - c2inv
    
    print("Equation of graph : %2.3f x^2 + %2.3f xy + %2.3fy^2  + %2.3fx + %2.3fy + %2.3f = 0" % (C[0][0], C[0][1]+C[1][0], C[1][1], A[0]+B[0], A[1] + B[1], -r+q))
    

    x = np.linspace(-3.0, 3.0, 200)
    y = np.linspace(-1.0, 5.0, 200)
    X, Y = np.meshgrid(x,y)

    #Graph Function
    F = C[0][0] * X * X + (C[0][1] + C[1][0]) * X * Y + C[1][1] * Y * Y + (A[0] + B[0])* X + (A[1] + B[1]) * Y - r + q
    plt.contour(X,Y,F,[0])
    plt.show()

driver()