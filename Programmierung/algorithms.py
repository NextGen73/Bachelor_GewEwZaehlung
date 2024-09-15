import numpy as np

# inflation parameter von g(z)
alpha = 0.1
# Schrittweite Differenzenverfahren
h = 0.01
# Schrittweite Gradientenverfahren
schrittGrad = 0.05
# startIntervall
lamA = 0
# endeIntervall
lamB = 1
# Mittelpunkt Kreis
gamma = 0.5
# Radius Kreis
r = .5
# Bedingungen, die fuer Werte gelten muessen, das ist nur ein Platzhalter
bedingung = np.zeros(1)
# Differenzenverfahren
diffVerfahren = "vorwaerts"

# muss zu Beginn aufgerufen werden, damit sinnvolle Ergebnisse berechnet werden koenenn
# man koennte es auch immer mit in der Funktion uebergeben, aber so ist es uebersichtlicher
def init(schrittweiteDifferenzen, schrittweiteGradienten, startIntervall, endeIntervall, bedingungen, differenzenVerfahren):
    """
    :param schrittweiteDifferenzen: Schrittweite, aufgrund deren der Differenzenquotient gebildet wird.
    :param schrittweiteGradienten: Faktor, mit dem die berechnete Richtung multipliziert wird.
    :param startIntervall: gibt untere Grenze des vorgegebenen Intervalls an.
    :param endeIntervall: gibt obere Grenze des vorgegebenen Intervalls an.
    :param bedingungen: gibt die gültigen Bedingungen an, die an das System gestellt werden.
    :param differenzenVerfahren: gibt an, ob Vorwaertsdifferenz oder symmetrische Differenz verwendet werden soll.
    """
    global h
    global schrittGrad
    global lamA
    global lamB
    global gamma
    global r
    global bedingung
    global diffVerfahren
    h = schrittweiteDifferenzen
    schrittGrad = schrittweiteGradienten
    lamA = startIntervall
    lamB = endeIntervall
    gamma = (startIntervall+endeIntervall)/2
    r = (endeIntervall-startIntervall)/2
    bedingung = bedingungen
    diffVerfahren = differenzenVerfahren

# verändert nur die Schrittweite, die beim Gradientenverfahren verwendet wird
def schrittweiteGradAendern(neueSchrittweite):
    global schrittGrad
    schrittGrad = neueSchrittweite

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
def schrittGradientenverfahren_festeSchrittweite(nablaF, x):
    """
    :param nableF: Ableitung der Funktion, die minimiert werden soll.
    :param x: Startwert fuer das Gradientenverfahren.
    :param schritt: Schrittweite, wie weit der Schritt des Gradientenverfahrens in die berechnete Richtung geht.
    :return: berechnete neue Stelle `x`.
    """

    abl_real = nablaF(x).real
    return x-schrittGrad*abl_real

# das beruht auf echter Trapezregel, wo man Mittelwert bildet und Differenz der Stellen
def quadratureContourIntegralCircleTrapez(f, m:int, s) -> complex:    
    # Berechne Stuetzstellen
    z = np.array([gamma+r*np.exp(2j*np.pi*k/m) for k in range(m+1)])
    return sum((f(z[k], s)+f(z[k+1], s)*np.exp(2*np.pi*1j/m))*np.exp(2*np.pi*1j*k/m) for k in range(m))*r*np.pi*1j/m

# das ist eher die Mittelpunktsregel, wo man Funktion nur an einer Stelle auswerten muss, Differenz wurde explizit berechnet und rausgezogen
def quadratureContourIntegralCircleMittelpunkt(f, m:int, s) -> complex:
    # Berechne Stuetzstellen
    zs = np.array([gamma+r*np.exp(2*np.pi*1j/m*(k+1/2)) for k in range(m)])
    return sum(f(zs[i], s)*np.exp(2*np.pi*1j*i/m) for i in range(m))*r*(np.exp(2*np.pi*1j/m)-1)

# minimiert die gewichtete Zaehlung der Eigenwerte, berechnet tatsaechliche gewichtete Eigenwert-Zaehlung, gibt alles aus
def EigenwerteMinimierenAufIntervall(M:np.ndarray, K:np.ndarray, startpunkt:np.ndarray, anzahlStuetzstellen:int, begrenzung:bool) -> np.ndarray:
    """
    :param M: Massematrix.
    :param K: Steifigkeitsmatrix.
    :anzahlStuetzstellen: Anzahl der Stuetzstellen in Quadraturformel.
    :return: Vektor, der berechnete Werte und Eigenwert-Zaehlungen enthaelt.
    """

    # damit sinnvolle Werte erkannt werden, muss bedingung vorher definiert werden
    # das ist durch Aufruf der Funktion algorithms.init() moeglich
    if(np.all(bedingung == 0)):
        raise ValueError("vorher init(...) aufrufen")

    len_s = len(startpunkt)

    # Gewichtungsfunktion g(z)
    def g(z: complex) -> complex:
        return -(z-((1+alpha)*lamA-alpha*lamB))*(z-((1+alpha)*lamB-alpha*lamA))
    
    # ist \1_{[\lamA,\lamB]}(z)
    def h(z:float)->float:
        if(z < lamB and z>lamA):
            return 1.
        return 0.

    # berechnet die gewichtete Anzahl der Eigenwerte mittels np.linalg.eigvals
    def J(s)->float:
        eigvals = np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s)))
        return sum(h(i)*g(i) for i in eigvals)

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
        
        return quadratureContourIntegralCircleTrapez(F, anzahlStuetzstellen, s)/2/np.pi/1j
    
    # Ableitung der approximierten gewichteten Ew-Zaehlung
    def nablaJ_Stern(s)->float:
        def nablaF(z:complex, s:np.ndarray)->np.ndarray[complex]:
            D = np.linalg.inv(z*M(s)-K(s))
            dMds = ableitungDurchDifferenz(M, s)
            dKds = ableitungDurchDifferenz(K, s)

            return g(z)*(np.trace(D.dot(dMds))-np.trace(((D.dot(z*dMds-dKds)).transpose(0,2,1).dot(D)).transpose(0,2,1)))
        return quadratureContourIntegralCircleTrapez(nablaF, anzahlStuetzstellen, s)/2/np.pi/1j

    # pruefe, ob Bedingungen von Wert erfuellt sind, sonst Projektion auf Intervallgrenze
    def bedingungenPruefen(x)->np.ndarray:
        for i in range(len(x)):
            if(x[i]<bedingung[i,0]):
                x[i] = bedingung[i,0]
            elif(x[i]>bedingung[i,1]):
                x[i] = bedingung[i,1]
        return x
    
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
        x_neu = schrittGradientenverfahren_festeSchrittweite(nablaJ_Stern, x_alt)
        # Bedingungen pruefen, evtl. Projektion
        x_neu = bedingungenPruefen(x_neu)
        # neues Tupel an result anhaengen, gleicher Aufbau wie oben, ab jetzt ist result wirklich eine 2d Matrix
        result = np.append(result, np.expand_dims(np.append(x_neu, [ungewichteteEwZaehlung_genau(x_neu), J(x_neu), J_Stern(x_neu)]), 1), axis=1)

        # falls begrenzung True ist, dann bricht ein Durchlauf nach 500 Schritten ab
        if(begrenzung):
            # nach 500 Durchlaeufen wird abgebrochen
            # nach dem ersten Durchlauf gilt: np.size(result,1)=2, da 2 Tupel (Anfangsdaten und Daten nach erstem Durchlauf) in result gespeichert wurden
            if(np.size(result,1) == 501):
                break

    return result