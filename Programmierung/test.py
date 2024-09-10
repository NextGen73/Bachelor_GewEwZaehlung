import numpy as np
import matplotlib.pyplot as plots
import algorithms
import cmath

lambda_a = 0.5
lambda_b = 1.5
alpha = 0.1
n: int = 8
s = np.array([3])
m = 1000
h = 0.01
z0Tilde = (lambda_a+lambda_b)/2
rTilde = (lambda_b-lambda_a)/2

phis = np.array([2*np.pi/m*(k+1/2) for k in range(m)])
# zs = np.array([z0Tilde+cmath.rect(rTilde, phi) for phi in phis])
zs = np.array([z0Tilde+rTilde*np.exp(2*np.pi*complex(0,1)/m*(k+1/2)) for k in range(m)])

cs=(zs-z0Tilde)/m

def K(s: float)->np.ndarray:
    return K_System1(s, 1.3, 3)

def M(s: float)->np.ndarray:
    return M_System1(3, 4, 1)

def K_System1(s, c, j):

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

def M_System1(m1, m2, j):

    result = np.zeros((n,n))
    diag = np.append(np.full(j-1, m1), np.full(n-j+1, m2))
    np.fill_diagonal(result, diag)
    return result

def D(s, z)-> np.ndarray:
    C = z*M(s)-K(s)
    return np.linalg.inv(C)

def g(z: complex) -> complex:
    return -(z-((1+alpha)*lambda_a-alpha*lambda_b))*(z-((1+alpha)*lambda_b-alpha*lambda_a))

# print(M(np.array([1,1])))
# x = np.array([0.0,0.0])
# print(np.trace(algorithms.Vorwaertsdifferenz(K, x, 0.1)))
# print(algorithms.Vorwaertsdifferenz(K, s, 0.001))

def JStern(s):
    return sum(g(zs[k])*np.trace(D(s, zs[k]).dot(M(s)))*cs[k] for k in range(m))/2/np.pi/complex(0,1)

def nablaJStern(s):
    result = 0
    for k in range(m):
        Dk = D(s, zs[k])
        dMds = algorithms.Vorwaertsdifferenz(M, s, h)
        dKds = algorithms.Vorwaertsdifferenz(K, s, h)
        result+=g(zs[k])*(np.trace(Dk.dot(dMds))-np.trace(Dk.dot(zs[k]*dMds-dKds).dot(Dk)))*cs[k]
    return result/2/np.pi/complex(0,1)

# print(JStern(s))
print(np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s))))
k = np.array([0.7581948, 1.19455199, 0.98672309])
print(sum(g(z) for z in k))