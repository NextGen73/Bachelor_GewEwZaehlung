import numpy as np

# inflation parameter g(z)
alpha = 0.1
# Schrittweite Differenzenverfahren
h = 0.01
# Schrittweite Gradientenverfahren
schrittGrad = 0.5
# Fehlertoleranz
eps= 1e-8
# startIntervall
lamA = 0
# endeIntervall
lamB = 1
# Mittelpunkt Kreis
gamma = 0.5
# Radius Kreis
r = .5
# Bedingungen, die fuer Werte gelten muessen, das ist nur ein Platzhalter
bedingung = np.empty

# muss zu Beginn aufgerufen werden, damit sinnvolle Ergebnisse berechnet werden koenenn
# man koennte es auch immer mit in der Funktion uebergeben, aber so ist es uebersichtlicher
def init(inflation, schrittweiteDifferenzen, schrittweiteGradienten, fehlertoleranz, startIntervall, endeIntervall, bedingungen):
    """
    :param inflation: Parameter alpha, wird in Gewichtungsfunktion verwendet.
    :param schrittweiteDifferenzen: Schrittweite, aufgrund deren der Differenzenquotient gebildet wird.
    :param schrittweiteGradienten: Schrittweite, die weit ein Schritt des Gradientenverfahrens geht.
    :param fehlertoleranz: Falls gewichtete EW-Zaehlung darunter faellt, wird die Minimierung beendet.
    """
    global alpha
    global h
    global schrittGrad
    global eps
    global lamA
    global lamB
    global gamma
    global r
    global bedingung
    alpha = inflation
    h = schrittweiteDifferenzen
    schrittGrad = schrittweiteGradienten
    eps = fehlertoleranz
    lamA = startIntervall
    lamB = endeIntervall
    gamma = (startIntervall+endeIntervall)/2
    r = (endeIntervall-startIntervall)/2
    bedingung = bedingungen

# approximiert Ableitung mittels Vorwaertsdifferenz
def  Vorwaertsdifferenz(f, x):
    """
    :param f: Funktion, deren Ableitung mithilfe der Vorwaertsdifferenz approximiert werden soll.
    :param x: Vektor als np.array, an dem Ableitung approximiert werden soll.
    :return: Approximierung von `f'(x)` mithilfe der Vorwaertsdifferenz.
    """

    # Laenge des Parameters ermitteln
    l=len(x)

    # zuerst wird erste partielle Ableitung approximiert, sollte immer vorhandne sein, solange x ein np.array ist
    xPlusH = x.copy()
    xPlusH[0] += h
    # partAbl speichert immer eine partielle Ableitung ab
    partAbl = (f(xPlusH)-f(x))/h

    # falls n>1, dann gibt es mehr als eine partielle Ableitung
    if l>1:
        # solange f ein np.array ausgibt, kann so bestimmt werden, wieviele Dimensionen es besitzt.
        # Sei A der Raum, in den f abbildet, dann ist die ausgegebene Matrix von der Form: A \times IR^l

        dimensionsF = np.ndim(partAbl)
        # hier wird nablaF, welches vorher ein Element in A war, auf ein Element in A \times IR erweitert,
        # damit auch die anderen partiellen Ableitungen geordnet in nablaF gespeichert werden koennen
        nablaF = np.expand_dims(partAbl, axis=dimensionsF)

        # wie oben fuer i=1 wird nun hier die i-te Ableitung bestimmt
        for i in range(1, l):
            xPlusH = x.copy()
            xPlusH[i] += h
            partAbl = (f(xPlusH)-f(x))/h
            # partAbl wird wie nablaF oben zu einem Element in A \times IR, um es A \times IR^i hinzuzufuegen
            # danach ist nableF ein Element in A \times IR^{i+1}
            nablaF = np.append(nablaF, np.expand_dims(partAbl, axis=dimensionsF) ,dimensionsF)
        # nach der Schleife liegt nablaF in der Form A \times IR^l vor, also genau in der Form, wie man es von der Ableitung einer Funktion f:IR^l -> A erwartet
    else:
        nablaF = partAbl
    return nablaF

# approximiert Ableitung mittels symmetrischer Differenz
def  symmetrischeDifferenz(f, x):
    """
    :param f: Funktion, deren Ableitung mithilfe der symmetrischen Differenz approximiert werden soll.
    :param x: Vektor als np.array, an dem Ableitung approximiert werden soll.
    :return: Approximierung von `f'(x)` mithilfe der Mitteldifferenz.
    """
    # Laenge des Parameters ermitteln
    l=len(x)

    # zuerst wird erste partielle Ableitung approximiert, sollte immer vorhandne sein, solange x ein np.array ist
    xPlusHHalbe = x.copy()
    xPlusHHalbe[0] += h/2

    xMinusHHalbe = x.copy()
    xMinusHHalbe[0] -= h/2
    # partAbl speichert immer eine partielle Ableitung ab
    partAbl = (f(xPlusHHalbe)-f(xMinusHHalbe))/h

    # falls n>1, dann gibt es mehr als eine partielle Ableitung
    if l>1:
        # solange f ein np.array ausgibt, kann so bestimmt werden, wieviele Dimensionen es besitzt.
        # Sei A der Raum, in den f abbildet, dann ist die ausgegebene Matrix von der Form: A \times IR^l

        dimensionsF = np.ndim(partAbl)
        # hier wird nablaF, welches vorher ein Element in A war, auf ein Element in A \times IR erweitert,
        # damit auch die anderen partiellen Ableitungen geordnet in nablaF gespeichert werden koennen
        nablaF = np.expand_dims(partAbl, axis=dimensionsF)

        # wie oben fuer i=1 wird nun hier die i-te Ableitung bestimmt
        for i in range(1, l):
            # wie oben nun fuer i-te part. Abl
            xPlusHHalbe = x.copy()
            xPlusHHalbe[i] += h/2

            xMinusHHalbe = x.copy()
            xMinusHHalbe[i] -= h/2

            partAbl = (f(xPlusHHalbe)-f(xMinusHHalbe))/h
            # partAbl wird wie nablaF oben zu einem Element in A \times IR, um es A \times IR^i hinzuzufuegen
            # danach ist nableF ein Element in A \times IR^{i+1}
            nablaF = np.append(nablaF, np.expand_dims(partAbl, axis=dimensionsF) ,dimensionsF)
        # nach der Schleife liegt nablaF in der Form A \times IR^l vor, also genau in der Form, wie man es von der Ableitung einer Funktion f:IR^l -> A erwartet
    else:
        nablaF = partAbl
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
    zs = np.array([gamma+r*np.exp(2*np.pi*1j/m*(k+1/2)) for k in range(m+1)])
    return sum((f(zs[i+1], s)+f(zs[i], s))/2*(zs[i+1]-zs[i]) for i in range(m))

# das ist eher die Mittelpunktsregel, wo man Funktion nur an einer Stelle auswerten muss, Differenz wurde explizit berechnet und rausgezogen
def quadratureContourIntegralCircleMittelpunkt(f, m:int, s) -> complex:
    # Berechne Stuetzstellen
    zs = np.array([gamma+r*np.exp(2*np.pi*1j/m*(k+1/2)) for k in range(m)])
    return sum(f(zs[i], s)*np.exp(2*np.pi*1j*i/m) for i in range(m))*r*(np.exp(2*np.pi*1j/m)-1)

# minimiert die gewichtete Zaehlung der Eigenwerte, berechnet tatsaechliche gewichtete Eigenwert-Zaehlung, gibt alles aus
def EigenwerteMinimierenAufIntervall(M:np.ndarray, K:np.ndarray, startpunkt:np.ndarray, anzahlStuetzstellen:int) -> np.ndarray:
    """
    :param M: Massematrix.
    :param K: Steifigkeitsmatrix.
    :anzahlStuetzstellen: Anzahl der Stuetzstellen in Quadraturformel.
    :return: Vektor, der berechnete Werte und Eigenwert-Zaehlungen enthaelt.
    """

    # damit sinnvolle Werte erkannt werden, muss bedingung vorher definiert werden
    # das ist durch Aufruf der Funktion algorithms.init() moeglich
    if(bedingung == np.empty):
        ValueError("vorher init(...) aufrufen")

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
        return sum(g(i) for i in eigvals)

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
            dMds = Vorwaertsdifferenz(M, s)
            dKds = Vorwaertsdifferenz(K, s)

            return g(z)*(np.trace(D.dot(dMds))-np.trace(((D.dot(z*dMds-dKds)).transpose(0,2,1).dot(D)).transpose(0,2,1)))
        return quadratureContourIntegralCircleTrapez(nablaF, anzahlStuetzstellen, s)/2/np.pi/1j

    # pruefe, ob Bedingungen von Wert erfuellt sind, sonst Projektion auf Intervallgrenze
    def bedingungenPruefen(x)->np.ndarray:
        for i in range(3):
            if(x[i]<bedingung[i,0]):
                x[i] = bedingung[i,0]
            elif(x[i]>bedingung[i,1]):
                x[i] = bedingung[i,1]
        return x
################## hier wird in jedem Schritt mehrmals M(s) und K(s) berechnet ##################

    # result wird spaeter eine 2d-Matrix, jetzt ist es noch ein Vektor
    # result[0:j,i] ist Wert, der in i-tem Schritt berechnet wurde
    # result[-3,i] ungewichtete Ew-Zaehlung mit Wert aus i-tem Schritt
    # result[-2,i] gewichtete Ew-Zaehlung mit Wert aus i-tem Schritt
    # result[-1,i] approximierte gewichtete Ew-Zaehlung mit Wert aus i-tem Schritt
    result = np.expand_dims(np.append(startpunkt, [ungewichteteEwZaehlung_genau(startpunkt), J(startpunkt), J_Stern(startpunkt)]), 1)

    # fuehre weiteren Schritt des Minimierungsverfahrens aus, wenn approximierte gew. Ew-Zaehlung nicht klein genug
    while result[-1,-1]>=eps:
        # neuen Wert s berechnen
        x_neu = schrittGradientenverfahren_festeSchrittweite(nablaJ_Stern, np.array(result[0:len_s,-1]))
        # Bedingungen pruefen, evtl. Projektion
        x_neu = bedingungenPruefen(x_neu)
        # neues Tupel an result anhaengen, gleicher Aufbau wie oben, ab jetzt ist result wirklich eine 2d Matrix
        result = np.append(result, np.expand_dims(np.append(x_neu, [ungewichteteEwZaehlung_genau(startpunkt), J(x_neu), J_Stern(x_neu)]), 1), axis=1)

        # gibt aller hundert Durchlaeufe eine Ausgabe, wie groß der genaue gewichtete Eigenwert ist, bei 500 Durchlaeufen wird abgebrochen
        # da aber nach dem ersten Durchlauf gilt: np.size(result,1)=2, da 2 Tupel in result gespeichert wurden, ist die erste Meldung bei dem 99. Druchlauf
        # nach 500 Durchlaeufen wird abgebrochen
        if(np.size(result,1)%100 == 0):
            print(result[-2,-1])
            if(np.size(result,1) == 501):
                break

    return result