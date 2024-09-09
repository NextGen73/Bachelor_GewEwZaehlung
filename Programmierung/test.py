import numpy as np
import matplotlib.pyplot as plots
import algorithms

def K(s):
    return K_System1(s, 8, 1.3, 3)

def M(s):
    return M_System1(8, 3, 4, 1)

def K_System1(s, n, c, j):

    result = np.zeros((n,n))

    diag=np.zeros(n)
    diag[range(j)] = 2*s
    diag[j] = s+c
    diag[range(j+1,n)] = 2*c 

    np.fill_diagonal(result, diag)

    nebendiag = np.zeros(n-1)
    nebendiag[range(j)] = -s
    nebendiag[range(j,n-1)] = -c

    result += np.diag(nebendiag, 1)
    result += np.diag(nebendiag, -1)
    return result

def M_System1(n, m1, m2, j):

    result = np.zeros((n,n))
    diag = np.append(np.full(j-1, m1), np.full(n-j+1, m2))
    np.fill_diagonal(result, diag)
    return result

# print(M(np.array([1,1])))
# x = np.array([0.0,0.0])
# print(x[0])
# print(np.trace(algorithms.Vorwaertsdifferenz(K, x, 0.1)))
print(len(np.array([0])))
print(algorithms.Vorwaertsdifferenz(K, np.array([0]), 0.001))
