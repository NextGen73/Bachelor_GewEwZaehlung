import numpy as np
import matplotlib.pyplot as plots
import algorithms

def M(s):
    if(len(s)!=2):
        return null
    a=1
    b=4
    return np.array([[a,a],[a,0]])+np.array([[b,b],[b,a]])*s[0]+np.array([[0,1],[1,b]])*s[1]

# print(M(np.array([1,1])))
x = np.array([0.0,0.0])
# print(x[0])
print(algorithms.Vorwaertsdifferenz(M, x, 0.1))