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
# Bedingungen, die für Werte gelten müssen, das ist nur ein Platzhalter
bedingung = np.empty

def init(inflation, schrittweiteDifferenzen, schrittweiteGradienten, fehlertoleranz, startIntervall, endeIntervall, bedingungen):
    """
    :param inflation: Parameter alpha, wird in Gewichtungsfunktion verwendet.
    :param schrittweiteDifferenzen: Schrittweite, aufgrund deren der Differenzenquotient gebildet wird.
    :param schrittweiteGradienten: Schrittweite, die weit ein Schritt des Gradientenverfahrens geht.
    :param fehlertoleranz: Falls gewichtete EW-Zaehlung darunter fällt, wird die Minimierung beendet.
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

def  Vorwaertsdifferenz(f, x):
    """
    :param f: Funktion, deren Ableitung mithilfe der Vorwaertsdifferenz approximiert werden soll.
    :param x: Vektor als np.array, an dem Ableitung approximiert werden soll.
    :raise ValueError: Wenn `h <= 0`.
    :return: Approximierung von `f'(x)` mithilfe der Vorwaertsdifferenz.
    """

    # Länge des Parameters ermitteln
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
        # damit auch die anderen partiellen Ableitungen geordnet in nablaF gespeichert werden können
        nablaF = np.expand_dims(partAbl, axis=dimensionsF)

        # wie oben für i=1 wird nun hier die i-te Ableitung bestimmt
        for i in range(1, l):
            xPlusH = x.copy()
            xPlusH[i] += h
            partAbl = (f(xPlusH)-f(x))/h
            # partAbl wird wie nablaF oben zu einem Element in A \times IR, um es A \times IR^i hinzuzufügen
            # danach ist nableF ein Element in A \times IR^{i+1}
            nablaF = np.append(nablaF, np.expand_dims(partAbl, axis=dimensionsF) ,dimensionsF)
        # nach der Schleife liegt nablaF in der Form A \times IR^l vor, also genau in der Form, wie man es von der Ableitung einer Funktion f:IR^l -> A erwartet
    else:
        nablaF = partAbl
    return nablaF

def  Mitteldifferenz(f, x):
    """
    :param f: Funktion, deren Ableitung mithilfe der Mitteldifferenz approximiert werden soll.
    :param x: Vektor als np.array, an dem Ableitung approximiert werden soll.
    :raise ValueError: Wenn `h <= 0`.
    :return: Approximierung von `f'(x)` mithilfe der Mitteldifferenz.
    """
    # Länge des Parameters ermitteln
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
        # damit auch die anderen partiellen Ableitungen geordnet in nablaF gespeichert werden können
        nablaF = np.expand_dims(partAbl, axis=dimensionsF)

        # wie oben für i=1 wird nun hier die i-te Ableitung bestimmt
        for i in range(1, l):
            # wie oben nun für i-te part. Abl
            xPlusHHalbe = x.copy()
            xPlusHHalbe[i] += h/2

            xMinusHHalbe = x.copy()
            xMinusHHalbe[i] -= h/2

            partAbl = (f(xPlusHHalbe)-f(xMinusHHalbe))/h
            # partAbl wird wie nablaF oben zu einem Element in A \times IR, um es A \times IR^i hinzuzufügen
            # danach ist nableF ein Element in A \times IR^{i+1}
            nablaF = np.append(nablaF, np.expand_dims(partAbl, axis=dimensionsF) ,dimensionsF)
        # nach der Schleife liegt nablaF in der Form A \times IR^l vor, also genau in der Form, wie man es von der Ableitung einer Funktion f:IR^l -> A erwartet
    else:
        nablaF = partAbl
    return nablaF

def schrittGradientenverfahren_festeSchrittweite(nablaF, x):
    """
    :param nableF: Ableitung der Funktion, die minimiert werden soll.
    :param x: Startwert für das Gradientenverfahren.
    :param schritt: Schrittweite, wie weit der Schritt des Gradientenverfahrens in die berechnete Richtung geht.
    :return: berechnete neue Stelle `x`.
    """

    abl_real = nablaF(x).real
    return x-schrittGrad*abl_real

# das beruht auf echter Trapezregel, wo man Mittelwert bildet und Differenz der Stellen
def quadratureContourIntegralCircleTrapez(f, m:int, s) -> complex:
    # Berechne Stützstellen
    zs = np.array([gamma+r*np.exp(2*np.pi*1j/m*(k+1/2)) for k in range(m+1)])
    return sum((f(zs[i+1], s)+f(zs[i], s))/2*(zs[i+1]-zs[i]) for i in range(m))

# das ist eher die Mittelpunktsregel, wo man Funktion nur an einer Stelle auswerten muss, Differenz wurde explizit berechnet und rausgezogen
def quadratureContourIntegralCircleMittelpunkt(f, m:int, s) -> complex:
    # Berechne Stützstellen
    zs = np.array([gamma+r*np.exp(2*np.pi*1j/m*(k+1/2)) for k in range(m)])
    return sum(f(zs[i], s)*np.exp(2*np.pi*1j*i/m) for i in range(m))*r*(np.exp(2*np.pi*1j/m)-1)

# minimiert die gewichtete Zaehlung der Eigenwerte, berechnet tatsächliche gewichtete Eigenwert-Zaehlung, gibt alles aus
def EigenwerteMinimierenAufIntervall(M:np.ndarray, K:np.ndarray, startpunkt:np.ndarray, anzahlStuetzstellen:int) -> np.ndarray:
    """
    :param M: Massematrix.
    :param K: Steifigkeitsmatrix.
    :param lamA: untere Grenze des Intervalls.
    :param lamB: obere Grenze des Intervalls.
    :anzahlStuetzstellen: Anzahl der Stützstellen in Quadraturformel.
    :param schrittweiteGradVerf: Schrittweite, wie weit ein Schritt des Gradientenverfahrens in die berechnete Richtung geht.
    :param fehlertoleranz: Grenze, wenn J(s) drunter fällt, dann wird Minimierungsverfahren beendet.
    :return: Vektor, 1. Spalte: berechneten Werte s, 2. Spalte: gewichtete Zählung (genau), 3. Spalte: gewichtete Zählung (approximiert).
    """

    len_s = len(startpunkt)

    # Gewichtungsfunktion g(z)
    def g(z: complex) -> complex:
        return -(z-((1+alpha)*lamA-alpha*lamB))*(z-((1+alpha)*lamB-alpha*lamA))
    
    # berechnet die Eigenwerte mittels funktion np.linalg.eigvals, siehe Definition mu
    def gewichteteEWZaehlung_genau(s)->float:
        # ist \1_{[\lamA,\lamB]}(z)
        def h(z:float)->float:
            if(z < lamB and z>lamA):
                return 1.
            return 0.
        
        eigvals = np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s)))
        return sum(h(i)*g(i) for i in eigvals)

    def J_Stern(s)->float:
        # zur besseren Lesbarkeit wird B = (zM-K)^{-1} M ausgelagert
        def B(z:complex, s:np.ndarray)-> np.ndarray[complex]:
            D = np.linalg.inv(z*M(s)-K(s))
            return g(z)*np.trace(D.dot(M(s)))
        
        return quadratureContourIntegralCircleTrapez(B, anzahlStuetzstellen, s)/2/np.pi/1j
    
    def nablaJ_Stern(s)->float:
        def nablaB(z:complex, s:np.ndarray)->np.ndarray[complex]:
            D = np.linalg.inv(z*M(s)-K(s))
            dMds = Vorwaertsdifferenz(M, s)
            dKds = Vorwaertsdifferenz(K, s)

            return g(z)*(np.trace(D.dot(dMds))-np.trace(((D.dot(z*dMds-dKds)).transpose(0,2,1).dot(D)).transpose(0,2,1)))
        return quadratureContourIntegralCircleTrapez(nablaB, anzahlStuetzstellen, s)/2/np.pi/1j

################## hier wird in jedem Schritt mehrmals M(s) und K(s) berechnet ##################

    # result wird eine 2d-Matrix, result[0:j-1,i] ist Wert, der in i-tem Schritt berechnet wurde
    # result[-2,i] genaue gewichtete Zaehlung der Eigenwerte mit Wert aus i-tem Schritt
    # result[-1,i] approximierte gewichtete Zaehlung der Eigenwerte mit Wert aus i-tem Schritt

    result = np.expand_dims(np.append(startpunkt, [gewichteteEWZaehlung_genau(startpunkt), J_Stern(startpunkt)]), 1)

    while result[-2,-1]>=eps:
        x_neu = schrittGradientenverfahren_festeSchrittweite(nablaJ_Stern, np.array(result[0:len_s,-1]))

        for i in range(3):
            if(x_neu[i]<bedingung[i,0]):
                x_neu[i] = bedingung[i,0]
            elif(x_neu[i]>bedingung[i,1]):
                x_neu[i] = bedingung[i,1]

        result = np.append(result, np.expand_dims(np.append(x_neu, [gewichteteEWZaehlung_genau(x_neu), J_Stern(x_neu)]), 1), axis=1)

        if(np.size(result,1)%100 == 0):
            print(result[-2,-1])
            if(np.size(result,1) == 500):
                break

    return result