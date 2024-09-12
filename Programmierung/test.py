import numpy as np
import matplotlib.pyplot as plots
import algorithms

n = 8
m = 100
j = int(n/2)
s = np.ones(j)
lambda_a = 1
lambda_b = 2

# hiermit ist gemeint, ob das erste oder das zweite System der Ausarbeitung untersucht wird
ausgewaehltesSystem = 2

def initAlgorithmen():
    global s
    global lambda_a
    global lambda_b
    if(ausgewaehltesSystem == 1):
        lambda_a = 1.5
        lambda_b = 3.0
        s = np.array([1])
        bedingungen1 = np.array([[0,3]])
        algorithms.init(0.1, 0.1, 0.5, 1e-6, lambda_a, lambda_b, bedingungen1)
    else:
        lambda_a = 0.9
        lambda_b = 1.2
        s = np.concatenate((np.full(j-1, 1.0), [2.5]))
        bedingungen2 = np.concatenate((np.tile(np.array([0.3,1.3]),j-1),np.array([1.5,3.5])))
        bedingungen2 = np.reshape(bedingungen2, (j,2))

        algorithms.init(0.1, 0.1, 0.5, 1e-6, lambda_a, lambda_b, bedingungen2)

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
        j = len(s)

        result = np.zeros((n,n))

        diagTeil1 = np.array([s[i]+s[i+1] for i in range(j-2)])
        diagTeil2 = np.full(n-j+1,1.5)
        diag = np.concatenate((diagTeil1, [s[j-2]+0.75], diagTeil2), axis=0)
    
        nebendiag = np.append(np.array([-s[i] for i in range(1,j-1)]), np.full(n-j+1, -0.75))

        np.fill_diagonal(result, diag)

        result += np.diag(nebendiag, 1)
        result += np.diag(nebendiag, -1)

        return result

    if(ausgewaehltesSystem == 1):
        return K_System1(s.real, 1.3, 3)
    return K_System2(s.real)

def M(s: float)->np.ndarray:
    def M_System1(s:np.ndarray, m1, m2, j):
        result = np.zeros((n,n))
        diag = np.append(np.full(j-1, m1*s[0]), np.full(n-j+1, m2*s[0]))
        np.fill_diagonal(result, diag)
        return result

    def M_System2(s):
        result = np.zeros((n,n))
        j =len(s)
        diag = np.append(np.full(j-2, 2), np.full(n-j+2, s[j-1]))
        np.fill_diagonal(result, diag)
        return result
    
    if(ausgewaehltesSystem == 1):
        return M_System1(s.real, 3, 4, 1)
    return M_System2(s.real)

result = algorithms.EigenwerteMinimierenAufIntervall(M, K, s, m)

VerlaufS = result[0:len(s),:]
s=VerlaufS[:,-1]
print(s)
EWgenau = result[-2,:]
EWapprox = result[-1,:]
EWungewichtet = result[-3,:].real

anzSchritte = len(EWapprox)
schritte = range(anzSchritte)
# dieser Plot zeigt, wie sich die Eigenwert-Zaehlungen waehrend des Minimierungsverfahrens entwickeln
plots.title("Zählung der Eigenwerte auf dem Intervall ["+str(lambda_a)+","+str(lambda_b)+"]")
plots.plot(schritte, EWgenau.real, 'r-', label="genau, gewichtet")
plots.plot(schritte, EWapprox.real, 'g--', label="approx, gewichtet")
plots.plot(schritte, EWungewichtet, 'b-', label="genau, ungewichtet")
plots.legend()
plots.show()

# dieser Plot zeigt, wie sich die Eigenwerte waehrend der Minimierung veraendern
# eigenwerte = np.array([np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s))) for s in VerlaufS])

print("endgültige Verteilung der Eigenwerte: ",np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s))))