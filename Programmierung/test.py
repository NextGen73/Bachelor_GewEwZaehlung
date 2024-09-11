import numpy as np
import matplotlib.pyplot as plots
import algorithms

lambda_a = 0.5
lambda_b = 1.5
n: int = 8
s = np.array([8])
m = 100

def K(s: float)->np.ndarray:
    return K_System1(s, 1.3, 3)

def M(s: float)->np.ndarray:
    return M_System1(s, 3, 4, 1)

def K_System1(s:np.ndarray, c, j:int):

    result = np.zeros((n,n))

    diag=np.ones(n)*s[0].real
    diag[range(j)] += 2
    diag[j]+=c
    diag[range(j+1,n)] += 2*c 

    np.fill_diagonal(result, diag)

    nebendiag = np.ones(n-1)*-s[0].real
    nebendiag[range(j,n-1)] += -c

    result += np.diag(nebendiag, 1)
    result += np.diag(nebendiag, -1)
    return result

def M_System1(s:np.ndarray, m1, m2, j):

    result = np.zeros((n,n))
    diag = np.append(np.full(j-1, m1*s[0]), np.full(n-j+1, m2*s[0]))
    np.fill_diagonal(result, diag)
    return result

result = algorithms.EigenwerteMinimierenAufIntervall(M, K, s, lambda_a, lambda_b, m, 0.5, 1e-6)

s = result[0:len(s),:]
EWgenau = result[-2,:]
EWapprox = result[-1,:]

schritte = len(EWapprox)

plots.plot(range(schritte), EWgenau, label="genau")
# plots.plot(range(schritte), s.real, label="s")
plots.show()

print(np.linalg.eigvals(np.linalg.inv(M(s[-1])).dot(K(s[-1]))))

