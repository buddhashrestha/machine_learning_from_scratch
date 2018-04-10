
from sklearn.datasets import load_iris

data = load_iris()

from sklearn.model_selection import train_test_split


import numpy as np

X_train, X_test, y_train, y_test = train_test_split(data['data'],data['target'], random_state = 0)

z = zip(X_train,y_train)
print("Z train : ",z[0][1])