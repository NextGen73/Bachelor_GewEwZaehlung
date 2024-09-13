import numpy as np
import matplotlib.pyplot as plots
import algorithms
import math

# Dimension der Masse- und Steifigkeitsmatrix, damit auch Anzahl an Massen in System
n = 8
# Anzahl der Quadraturstellen
m = 100
# j gibt für System 1 an, ab welchem Index die Masse fest ist
# für System 2 gibt es die Länge des Vektors s an
j = int(n/2)
# Startwert der Minimierung
s = np.ones(j)
# untere Grenze des Intervalls
lambda_a = 1
# obere Grenze des Intervalls
lambda_b = 2

# hiermit ist gemeint, ob das erste oder das zweite System der Ausarbeitung untersucht wird
ausgewaehltesSystem = 2

# diese Funktion definiert abhängig von ausgewaehltesSystem den Startwert, das Intervall und die Bedingungen, die für s gelten sollen
# das untersuchte Intervall und die Bedingungen können durch Aufruf von algorithms.init() veraendert werden
# der Startwert s wird in der Funktion algorithms.EigenwerteMinimierenAufIntervall() uebergeben
def initAlgorithmen():
    global s
    global lambda_a
    global lambda_b

    if(ausgewaehltesSystem == 1):
        lambda_a = 1.5
        lambda_b = 2.5
        s = np.array([3.5])
        bedingungen1 = np.array([[2,5]])

        algorithms.init(0.1, 0.1, 0.5, 1e-6, lambda_a, lambda_b, bedingungen1, "vorwaerts")
    else:
        lambda_a = 0.9
        lambda_b = 1.5
        s = np.concatenate((np.full(j-1, 0.5), [1.0]))
        bedingungen2 = np.concatenate((np.tile(np.array([0.1,2.0]),j-1),np.array([.5,3.5])))
        bedingungen2 = np.reshape(bedingungen2, (j,2))

        algorithms.init(0.1, 0.1, 0.5, 1e-6, lambda_a, lambda_b, bedingungen2, "vorwaerts")

# berechnet K abhaenig vom verwendeten System
def K(s: float)->np.ndarray:
    def K_System1(s:np.ndarray):
        result = np.zeros((n,n))

        diag=np.concatenate((np.full(j-1,2*s[0]), [s[0]+1.5], np.full(n-j,3)))
        np.fill_diagonal(result, diag)

        nebendiag= np.concatenate((np.full(j-1,-s[0]), np.full(n-j, -1.5)))
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
        return K_System1(s.real)
    return K_System2(s.real)

def M(s: float)->np.ndarray:
    def M_System1(s:np.ndarray):
        result = np.zeros((n,n))
        diag = np.concatenate((np.full(j,4), np.full(n-j, s[0]+1)))
        np.fill_diagonal(result, diag)
        return result

    def M_System2(s):
        result = np.zeros((n,n))
        j =len(s)
        diag = np.append(np.full(j-2, 2), np.full(n-j+2, s[j-1]))
        np.fill_diagonal(result, diag)
        return result
    
    if(ausgewaehltesSystem == 1):
        return M_System1(s.real)
    return M_System2(s.real)

if __name__ == "__main__":
    
    initAlgorithmen()

    # Minimierungsverfahren auf Problem anwenden.
    # result enthält in jeder Spalte die zu einem Schritt zugehörigen Werte
    # zuerst kommt der berechnete Wert s, dann die ungewichtete Eigenwertzaehlung, die gewichtete Eigenwertzaehlung und zum Schluss die approximierte gewichtete Eigenwertzaehlung
    result = algorithms.EigenwerteMinimierenAufIntervall(M, K, s, m)

    verlaufS = np.transpose(result[0:len(s),:],axes=(1,0))
    EWgenau = result[-2,:]
    EWapprox = result[-1,:]
    EWungewichtet = result[-3,:].real

    anzSchritte = len(EWapprox)
    schritte = range(anzSchritte)

    colors=np.tile(['b', 'g', 'r', 'c', 'm'], math.ceil(n/5))
    eigenwerte = np.array([np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s))) for s in verlaufS])
    
    fig,(axo,axu) = plots.subplots(2, sharex=True)
    # dieser Plot zeigt, wie sich die Eigenwert-Zaehlungen waehrend des Minimierungsverfahrens entwickeln
    axo.set_title("Zählung der Eigenwerte auf dem Intervall ["+str(lambda_a)+","+str(lambda_b)+"]")
    axo.plot(schritte, EWgenau.real, 'r-', label="genau, gewichtet")
    axo.plot(schritte, EWapprox.real, 'g--', label="approx, gewichtet")
    # plots.plot(schritte, EWungewichtet, 'b-', label="genau, ungewichtet")
    axo.legend()

    # dieser Plot zeigt, wie sich die Eigenwerte waehrend der Minimierung veraendern
    axu.set_title("Entwicklung der Eigenwerte bezüglich ["+str(lambda_a)+","+str(lambda_b)+"]")
    # axu.yticks(np.arange(0,np.max(eigenwerte)+.1, 0.2))
    for i in range(n):
        verlaufEinEigenwert = eigenwerte[:,i]
        axu.plot(schritte, verlaufEinEigenwert, label="Ew "+str(i+1), color=colors[i], linewidth=0.5)

    axu.plot(schritte,np.full(anzSchritte,lambda_a), 'k')
    axu.plot(schritte,np.full(anzSchritte,lambda_b), 'k')

    axu.legend()
    plots.show()