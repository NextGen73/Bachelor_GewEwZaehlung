import numpy as np
import math
import matplotlib.pyplot as plots
import time

# all diese Werte werden in init() und systemAuswaehlen() neu definiert,
# sie stehen hier nur, damit es zu keinem Fehler kommt

# definiert das verwendete System
system = 1
# definiert Startwert für Minimierung
s = np.ones(1)
bedingung = np.array(1)
# definiert das Intervall
lambda_a=-1
lambda_b=1
# definiert Steifigkeits- und Massematrix
K = 1
M = 1
# definiert die verwendete Quadraturformel
quadratur = 1
# definiert die Anzahl an Teilintervallen
m = 10
# diese Variablen definieren den Kreis
r=1
gamma = 0
# defniert feste Schrittweite bei Schritt des Gradientenverfahrens
lambdaStern = 1
# definiert die maximale Anzahl an Schritten pro Durchlauf
begrenzung = 1


# muss zu Beginn aufgerufen werden, damit sinnvolle Ergebnisse berechnet werden koennen
# hier werden globale Variablen definiert, die ueberall benoetigt werden, aber normalerweise nur einmal definiert werden muessen
def init(quadraturformel, freiheitsgrade=8, hilfszahl=4, anzahlTeilintervalleQuadratur = 100, schrittweiteGradVerfahren = 0.05, inflationParameter=0.1, maxIter = 500, schrittweiteDiffVerfahren=0.1, differenzenVerfahren="vorwaerts"):
    global quadratur
    global n
    global j
    global m
    global h
    global alpha
    global diffVerfahren
    global begrenzung
    global lambdaStern
    quadratur = quadraturformel
    n = freiheitsgrade
    j = hilfszahl
    m = anzahlTeilintervalleQuadratur
    h = schrittweiteDiffVerfahren
    alpha = inflationParameter
    begrenzung = maxIter
    diffVerfahren = differenzenVerfahren
    lambdaStern = schrittweiteGradVerfahren

# hier kann man auswaehlen, welches System verwendet werden soll
# dadurch wird auch der Startwert, das Intervall, die Bedingungen und auch die verwendeten Matrizen definiert
def systemAuswaehlen(ausgewaehltesSystem):
    global system
    global lambda_a
    global lambda_b
    global s
    global bedingungen
    global gamma
    global r
    global K
    global M

    def K_System1(s:np.ndarray):
        s = s.real
        result = np.zeros((n,n))

        diag=np.concatenate((np.full(j-1,2*s[0]), [s[0]+1.5], np.full(n-j,3)))
        np.fill_diagonal(result, diag)

        nebendiag= np.concatenate((np.full(j-1,-s[0]), np.full(n-j, -1.5)))
        result += np.diag(nebendiag, 1)
        result += np.diag(nebendiag, -1)

        return result

    def K_System2(s:np.ndarray):
        s = s.real        
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

    def M_System1(s:np.ndarray):
        s = s.real
        result = np.zeros((n,n))
        diag = np.concatenate((np.full(j,4), np.full(n-j, s[0]+1)))
        np.fill_diagonal(result, diag)
        return result

    def M_System2(s:np.ndarray):
        s = s.real
        result = np.zeros((n,n))
        j =len(s)
        diag = np.append(np.full(j-2, 2), np.full(n-j+2, s[j-1]))
        np.fill_diagonal(result, diag)
        return result

    if(ausgewaehltesSystem == 1):
        system = 1
        lambda_a = 1.5
        lambda_b = 2.5
        s = np.array([3.5])
        bedingungen = np.array([[2,5]])
        # definiert K und M abhaengig vom verwendeten System
        K = K_System1
        M = M_System1
    else:
        system = 2
        lambda_a = 0.9
        lambda_b = 1.5
        s = np.concatenate((np.full(j-1, 0.7), [1.5]))
        bedingungen = np.concatenate((np.tile(np.array([0.1,2.0]),j-1),np.array([.5,3.5])))
        bedingungen = np.reshape(bedingungen, (j,2))
        # definiert K und M abhaengig vom verwendeten System
        K = K_System2
        M = M_System2

    gamma = (lambda_a+lambda_b)*0.5
    r = (lambda_b-lambda_a)*0.5

# das beruht auf echter Trapezregel, wo man Mittelwert bildet und Differenz der Stellen
def quadratureContourIntegralCircleTrapez(f, s, m) -> complex:    
    # Berechne Stuetzstellen
    z = np.array([gamma+r*np.exp(2j*np.pi*k/m) for k in range(m+1)])
    return sum((f(z[k], s)+f(z[k+1], s)*np.exp(2*np.pi*1j/m))*np.exp(2*np.pi*1j*k/m) for k in range(m))*r*np.pi*1j/m

# das beruht auf verschobener Trapezregel, wo man Mittelwert bildet und Differenz der Stellen
def quadratureContourIntegralCircleTrapezNeu(f, s, m) -> complex:    
    # Berechne Stuetzstellen
    z = np.array([gamma+r*np.exp(2j*np.pi*(k+0.5)/m) for k in range(m+1)])
    return sum((f(z[k], s)+f(z[k+1], s)*np.exp(2*np.pi*1j/m))*np.exp(2*np.pi*1j*k/m) for k in range(m))*r*np.pi*1j/m

# das ist eher die Mittelpunktsregel, wo man Funktion nur an einer Stelle auswerten muss, Differenz wurde explizit berechnet und rausgezogen
def quadratureContourIntegralCircleMittelpunkt(f, s, m) -> complex:
    # Berechne Stuetzstellen
    zs = np.array([gamma+r*np.exp(2*np.pi*1j/m*(k+1/2)) for k in range(m)])
    return sum(f(zs[k], s)*np.exp(2*np.pi*1j*(k+1/2)/m) for k in range(m))*2*np.pi*1j*r/m

# diese Quadraturformel ist optimal für 2 Stuetzstellen, siehe Gauss 2 Punkt Formel
def quadratureContourIntegralCircleGaussZwei(f, s, m) -> complex:
    #Stuetzstellen sind hier ein mx2 Array
    PiIDurchM = np.pi*1j/m
    zs = gamma+r*np.exp(PiIDurchM*np.array([[2*k+1-1/math.sqrt(3), 2*k+1+1/math.sqrt(3)] for k in range(m)]))
    return sum((f(zs[k,0], s)*np.exp(-PiIDurchM/math.sqrt(3))+f(zs[k,1], s)*np.exp(PiIDurchM/math.sqrt(3)))*np.exp(PiIDurchM*(2*k+1)) for k in range(m))*PiIDurchM*r

# approximiert Ableitung mittels Differenzenverfahren
def ableitungDurchDifferenz(f, x):
    """
    :param f: Funktion, deren Ableitung mithilfe des Differenzenverfahren approximiert werden soll.
    :param x: Vektor als np.array, an dem Ableitung approximiert werden soll.
    :return: Approximierung von `f'(x)` mithilfe eines Differenzenverfahrens.
    """
    # Laenge des Arguments ermitteln
    l=len(x)

    # solange f ein np.array ausgibt, kann so bestimmt werden, wieviele Dimensionen es besitzt.
    dimensionsF = np.ndim(f(x))
    # nablaF wird als leere Matrix initialisiert, da die Maße noch nicht klar sind
    nablaF = np.empty
    # es wird für jedes Element von x die partielle Ableitung approximiert
    for i in range(0, l):
        if (diffVerfahren=="vorwaerts"):
            # für Vorwaertsdifferenz wird Ableitung durch (f(x+h)-f(x))/h approximiert
            xPlusH = x.copy()
            xPlusH[i] += h

            partAbl = (f(xPlusH)-f(x))/h
        else:
            # für symmetrische Differenz wird Ableitung durch (f(x+h/2)-f(x-h/2))/h approximiert
            xPlusHHalbe = x.copy()
            xPlusHHalbe[i] += h/2
            xMinusHHalbe = x.copy()
            xMinusHHalbe[i] -= h/2

            partAbl = (f(xPlusHHalbe)-f(xMinusHHalbe))/h
        # für i>0 wird nableF durch die (i+1)-te partielle Ableitung erweitert
        if(i>0):
            nablaF = np.append(nablaF, np.expand_dims(partAbl, axis=dimensionsF) ,dimensionsF)
        else:
            # für i=0 muss nablaF eine Matrix in A\times \R werden, damit die anderen partiellen Ableitungen mit angehaengt werden koennen
            nablaF = np.expand_dims(partAbl, axis=dimensionsF)
    # nach der Schleife liegt nablaF in der Form A \times IR^l vor, also genau in der Form, wie man es von der Ableitung einer Funktion f:IR^l -> A erwartet
    return nablaF

# ein Schritt des Gradientenverfahrens
def schrittGradientenverfahren(nablaF, x, lambdaStern=0.05, dynamisch=False):
    """
    :param nableF: Ableitung der Funktion, die minimiert werden soll.
    :param x: Startwert fuer das Gradientenverfahren.
    :param schrittGradFunktion: Schrittweite, wie weit der Schritt des Gradientenverfahrens in die berechnete Richtung geht.
    :return: berechnete neue Stelle `x`.
    """

    def punktDynamischBerechnen(nablaF, x, d):
        # kann nur eintreten, wenn das Intervall viel zu groß ist, sonst sollte vorher eine andere Bedingung eintreten
        x_alt = x
        for i in range(1000):
            # leider muss man aus Performance-Gründen die Schrittweite auf 0.5 erhöhen
            x_neu = x_alt-lambdaStern*d
            # falls ein Wert außerhalb der zulässigen Menge ist, so nehme diesen Wert als Ergebnis für das Gradientenverfahren
            # es würde nichts bringen noch weiter zu machen, da nachher der Wert immer auf den Rand der zulässigen Menge gesetzt
            if(np.any(x_neu != bedingungenPruefen(x_neu))):
                return x_neu
            # berechne es einmal, da es in den nächsten zwei Abfragen benötigt wird
            nablaFMalD = nablaF(x_neu).dot(d)
            # aufgrund von Ungenauigkeiten und der Diskretisierung ist es besser |nablaFMalD| < epsilon zu fordern
            # anstatt nablaFMalD = 0
            if(abs(nablaFMalD)<1e-5):
                return x_neu
            # falls nablaFMalD kleiner Null ist, dann ist man zu weit gelaufen
            # falls aber schon im ersten Schritt diese Bedingung eintritt, dann würde man in eine Endlosschleife laufen, da das Gradientenverfahren dann den selben Wert zurückgibt, wie reinkam
            # beim nächsten Durchlauf wären daher wieder alle Werte gleich und man würde in eine Endlosschleife laufen
            if(nablaFMalD < 0):
                return x_alt if i==1 else x_neu
            x_alt = x_neu
        return x_alt

    abl_real = nablaF(x).real

    if(dynamisch):
        return punktDynamischBerechnen(nablaF, x, abl_real)
    else:
        return x-lambdaStern*abl_real

# pruefe, ob Bedingungen von Wert erfuellt sind, sonst Projektion auf Intervallgrenze
def bedingungenPruefen(x)->np.ndarray:
    for i in range(len(x)):
        if(x[i]<bedingungen[i,0]):
            x[i] = bedingungen[i,0]
        elif(x[i]>bedingungen[i,1]):
            x[i] = bedingungen[i,1]
    return x
    
# minimiert die gewichtete Zaehlung der Eigenwerte, berechnet tatsaechliche gewichtete Eigenwert-Zaehlung, gibt alles zurueck
def EigenwerteMinimierenAufIntervall(startpunkt:np.ndarray, anzahlTeilintervalle:int, schrittweiteGrad=0.05, dynamischSchritt=False, approxNablaJ=False) -> np.ndarray:
    """
    :param startpunkt: Ausgangpunkt für Minimierung.
    :anzahlTeilintervalle: Anzahl der Stuetzstellen in Quadraturformel.
    :param begrenzung: Matrix, in der die Intervalle gespeichert sind, die der Design-Parameter annehmen darf.
    :param quadratur: Funktion, die eine Quadraturformel verwendet.
    :return: Vektor, der berechnete Werte und Eigenwert-Zaehlungen enthaelt.
    """

    # damit sinnvolle Werte erkannt werden, muss bedingungen vorher definiert werden
    # das ist durch Aufruf der Funktion algorithms.systemAuswaehlen() moeglich
    if(np.all(bedingungen == 0)):
        raise ValueError("vorher init(...) aufrufen")

    len_s = len(startpunkt)

    # Gewichtungsfunktion g(z)
    def g(z: complex) -> complex:
        return -(z-((1+alpha)*lambda_a-alpha*lambda_b))*(z-((1+alpha)*lambda_b-alpha*lambda_a))
    
    # ist \1_{[\lambda_a,\lambda_b]}(z)
    def h(z:float)->float:
        if(z < lambda_b and z>lambda_a):
            return 1.
        return 0.

    # berechnet die gewichtete Anzahl der Eigenwerte mittels np.linalg.eigvals
    def J(s)->float:
        eigvals = np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s)))
        return sum(h(i)*g(i) for i in eigvals)
    
    # berechnet NablaJ durch J und Differenzenverfahren
    def nablaJ(s)->float:
        return ableitungDurchDifferenz(J, s)

    #berechnet, wie viele Eigenwerte sich in dem vorgegebenen Intervall befinden
    def ungewichteteEwZaehlung_genau(s)->int:
        eigvals = np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s)))
        return sum(h(i) for i in eigvals)

    # approximiert die gewichtete Anzahl an Eigenwerten im vorgegbenen Intervall
    def J_Stern(s)->float:
        # zur besseren Lesbarkeit wird F = g(z)*tr((zM-K)^{-1} M) ausgelagert
        def F(z:complex, s:np.ndarray)-> np.ndarray[complex]:
            D = np.linalg.inv(z*M(s)-K(s))
            return g(z)*np.trace(D.dot(M(s)))
        
        return quadratur(F, s, anzahlTeilintervalle)/2/np.pi/1j
    
    # Ableitung der approximierten gewichteten Ew-Zaehlung
    def nablaJ_Stern(s)->float:
        def nablaF(z:complex, s:np.ndarray)->np.ndarray[complex]:
            D = np.linalg.inv(z*M(s)-K(s))
            dMds = ableitungDurchDifferenz(M, s)
            dKds = ableitungDurchDifferenz(K, s)

            return g(z)*(np.trace(D.dot(dMds))-np.trace(((D.dot(z*dMds-dKds)).transpose(0,2,1).dot(D)).transpose(0,2,1)))
        return quadratur(nablaF, s, anzahlTeilintervalle)/2/np.pi/1j

   
    # result wird spaeter eine 2d-Matrix, jetzt ist es noch ein Vektor
    # result[0:j,i] ist Wert, der in i-tem Schritt berechnet wurde
    # result[-3,i] ungewichtete Ew-Zaehlung mit Wert aus i-tem Schritt
    # result[-2,i] gewichtete Ew-Zaehlung mit Wert aus i-tem Schritt
    # result[-1,i] approximierte gewichtete Ew-Zaehlung mit Wert aus i-tem Schritt
    result = np.expand_dims(np.append(startpunkt, [ungewichteteEwZaehlung_genau(startpunkt), J(startpunkt), J_Stern(startpunkt)]), 1)

    # fuehre weiteren Schritt des Minimierungsverfahrens aus, wenn approximierte gew. Ew-Zaehlung nicht klein genug
    while result[-3,-1]>0:
        x_alt = np.array(result[0:len_s,-1])
        # neuen Wert s berechnen
        x_neu = schrittGradientenverfahren(nablaJ if approxNablaJ else nablaJ_Stern, x_alt, schrittweiteGrad, dynamischSchritt)
        # Bedingungen pruefen, evtl. Projektion
        x_neu = bedingungenPruefen(x_neu)
        # neues Tupel an result anhaengen, gleicher Aufbau wie oben, ab jetzt ist result wirklich eine 2d Matrix
        result = np.append(result, np.expand_dims(np.append(x_neu, [ungewichteteEwZaehlung_genau(x_neu), J(x_neu), J_Stern(x_neu)]), 1), axis=1)

        # nach einer gewissen Anzahl an Durchlaeufen wird abgebrochen
        # nach dem ersten Durchlauf gilt: np.size(result,1)=2, da 2 Tupel (Anfangsdaten und Daten nach erstem Durchlauf) in result gespeichert wurden
        # somit wird bei np.size()==begrenzung+1 abgebrochen, damit es genau `begrenzung` viele Schritte waren
        if(np.size(result,1) == begrenzung+1):
            break

    return result

# ruft EigenwerteMinimierenAufIntervall auf, stoppt benoetigte Zeit, fertigt Plots an und gibt Eckdaten aus
def minimierenPlottenUndEckdatenAnzeigen(anzahlTeilintervalle, schrittweiteGrad=0.05, dynamischSchritt=False, approxNablaJ=False):
    global m
    m = anzahlTeilintervalle
    
    # mit startzeit und später vergangeneZeit wird nur die Zeit gemessen, die die Funktion EigenwerteMinimierenAufIntervall benötigt,
    # also genau die Zeit, die das Minimierungsverfahren braucht
    startzeit = time.time()

    # Minimierungsverfahren auf Problem anwenden.
    # result enthält in jeder Spalte die zu einem Schritt zugehörigen Werte
    # zuerst kommt der berechnete Wert s, dann die ungewichtete Eigenwertzaehlung, die gewichtete Eigenwertzaehlung und zum Schluss die approximierte gewichtete Eigenwertzaehlung
    result = EigenwerteMinimierenAufIntervall(s, anzahlTeilintervalle, schrittweiteGrad, dynamischSchritt, approxNablaJ)
    # in vergangeneZeit wird die Zeit in Sekunden gespeichert, die das Minimierungsverfahren benötigte
    vergangeneZeit = time.time()-startzeit

    # gibt den Verlauf des Parameters s an, wird benötigt, um den Verlauf der Eigenwerte zu berechnen
    verlaufS = np.transpose(result[0:len(s),:],axes=(1,0))
    # die drei Arrays werden für den oberen Plot benötigt. Sie geben an, wie sich die Eigenwert-Zaehlungen verändert haben
    EWgenau = result[-2,:]
    EWapprox = result[-1,:]
    EWungewichtet = result[-3,:].real

    # anzSchritte und schritte werden definiert, damit die Plots eine gültige x-Achse erhalten
    anzSchritte = len(EWapprox)
    schritte = range(anzSchritte)

    # gibt die Farben des unteren Plots an, damit das Array lang genug ist, wird es oft genug mit sich selbst verkettet
    colors=np.tile(['b', 'g', 'r', 'c', 'm'], math.ceil(n/5))
    # wird für den unteren Plot benötigt, gibt den Verlauf aller Eigenwerte an
    eigenwerte = np.array([np.sort(np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s)))) for s in verlaufS])
    
    if(EWungewichtet[-1]==0):
        ergebnis = "ja"
    else:
        ergebnis = "nein"

    anzahlIterationen = len(EWungewichtet)-1

    # der angezeigt Plot wird aus einem oberen und unteren Plot bestehen
    fig,(axo,axu) = plots.subplots(2, sharex=True)
    # dieser Plot zeigt, wie sich die Eigenwert-Zaehlungen waehrend des Minimierungsverfahrens entwickeln
    axo.set_title("Zählung der Eigenwerte auf dem Intervall ["+str(lambda_a)+", "+str(lambda_b)+"]")
    axo.plot(schritte, EWgenau.real, 'r-', label="genau, gewichtet")
    axo.plot(schritte, EWapprox.real, 'g--', label="approx, gewichtet")
    axo.plot(schritte, EWungewichtet, 'b-', label="genau, ungewichtet")
    axo.legend()

    # dieser Plot zeigt, wie sich die Eigenwerte waehrend der Minimierung veraendern
    axu.set_title("Entwicklung der Eigenwerte bezüglich ["+str(lambda_a)+", "+str(lambda_b)+"]")
    for i in range(n):
        verlaufEinEigenwert = eigenwerte[:,i]
        axu.plot(schritte, verlaufEinEigenwert, label="Ew "+str(i+1), color=colors[i])

    axu.plot(schritte,np.full(anzSchritte,lambda_a), 'k')
    axu.plot(schritte,np.full(anzSchritte,lambda_b), 'k')
    # axu.legend()

    # Plot anzeigen
    plots.show()

    # Angabe der wichtigsten Eckdaten 
    print("Eckdaten für System "+str(system)+":")
    print("Verfahren brachte ein Ergebnis: "+ergebnis)
    print("Startwert Parameter: ",str(np.round(verlaufS[0].real, 3)))
    print("Endwert   Parameter: ", np.round(verlaufS[-1].real, 3))
    print("Intervall: ["+str(lambda_a)+", "+str(lambda_b)+"]")
    # print("Eigenwerte am Anfang: "+str(np.round(eigenwerte[0,:], 2)))
    # print("Eigenwerte am Ende:   "+str(eigenwerte[-1,:].round(2)))
    print("Anzahl Stützstellen Quadratur: "+str(m))
    print("Schrittweite des Gradientenverfahrens: "+str(schrittweiteGrad))
    print("Anzahl an benötigten Iterationen: "+str(anzahlIterationen))
    print("für Minimierung vergangene Zeit in s: "+str(np.round(vergangeneZeit, 2)))
    print("")