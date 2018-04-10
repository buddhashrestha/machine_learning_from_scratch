import operator
import numpy as np
def maxSubArray(a):
    x =[]
    x.append(a[0])
    for i in range(1,len(a)):
        x.append(max(a[i]+x[i-1],a[i]))

    index, value = max(enumerate(x), key=operator.itemgetter(1))
    return value

print(maxSubArray(np.array([-2, -3, 4, -1, -2, 1, 5, -3])))