import numpy as np
import matplotlib.pyplot as plots
import algorithms

n: int = 8
m = 140
s = np.empty
# hiermit ist gemeint, ob das erste oder das zweite System der Ausarbeitung untersucht wird
ausgewaehltesSystem = 2

def initAlgorithmen():
    global s
    if(ausgewaehltesSystem == 1):
        lambda_a1 = 1.5
        lambda_b1 = 3.0
        s = np.array([1])
        bedingungen1 = np.array([[0,3]])
        algorithms.init(0.1, 0.1, 0.5, 1e-6, lambda_a1, lambda_b1, bedingungen1)
    else:
        lambda_a2 = 1.5
        lambda_b2 = 3.0
        s = np.array([1,2,1], dtype=float)
        bedingungen2 = np.array([[0,3],[0,3],[0.75,1.75]])

        algorithms.init(0.1, 0.1, 0.5, 1e-6, lambda_a2, lambda_b2, bedingungen2)
        s = np.array([1,2,1], dtype=float)

initAlgorithmen()

def K(s: float)->np.ndarray:
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

    def K_System2(s:np.ndarray):
        result = np.zeros((10,10))

        diag = np.zeros(10)
        diag[:6] = 2*s[0].real
        diag[6] = (s[0]+s[1]).real
        diag[7:] = 2*s[1].real

        nebendiag = np.append(np.full(6, -s[0].real), np.full(3, -s[1].real))

        np.fill_diagonal(result, diag)

        result += np.diag(nebendiag, 1)
        result += np.diag(nebendiag, -1)

        return result

    if(ausgewaehltesSystem == 1):
        return K_System1(s, 1.3, 3)
    return K_System2(s)

def M(s: float)->np.ndarray:
    def M_System1(s:np.ndarray, m1, m2, j):
        result = np.zeros((n,n))
        diag = np.append(np.full(j-1, m1*s[0]), np.full(n-j+1, m2*s[0]))
        np.fill_diagonal(result, diag)
        return result

    def M_System2(s):
        result = np.zeros((10,10))
        diag = np.append(np.full(6, s[2].real), np.full(4,0.5))
        np.fill_diagonal(result, diag)
        return result
    
    if(ausgewaehltesSystem == 1):
        return M_System1(s, 3, 4, 1)
    return M_System2(s)

result = algorithms.EigenwerteMinimierenAufIntervall(M, K, s, m)

VerlaufS = result[0:len(s),:]
s=VerlaufS[:,-1]
print(s)
EWgenau = result[-2,:]
EWapprox = result[-1,:]

schritte = len(EWapprox)

plots.plot(range(schritte), EWgenau.real, label="genau")
plots.plot(range(schritte), EWapprox.real, label="approx")
plots.show()

print(np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s))))

# print(K(s))

# A = np.array([[[2,2],[3,2],[1,2]],[[1,0],[0,0],[1,1]],[[2,2],[3,2],[1,2]]])
# B = np.array([[3,1,2],[1,2,3],[1,2,3]])

# print((A.transpose(0,2,1).dot(B)).transpose(0,2,1))