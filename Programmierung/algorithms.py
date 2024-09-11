import numpy as np

alpha = 0.1
schrittweiteDiffverfahren = 0.01
lamA: float
lamB: float

def init(start, ende):
    lamA = start
    lamB = ende

def  Vorwaertsdifferenz(f, x, h):
    """
    :param f: Funktion, deren Ableitung mithilfe der Vorwaertsdifferenz approximiert werden soll.
    :param x: Vektor als np.array, an dem Ableitung approximiert werden soll.
    :param h: Schrittweite, aufgrund deren der Differenzenquotient gebildet wird.
    :raise ValueError: Wenn `h <= 0`.
    :return: Approximierung von `f'(x)` mithilfe der Vorwaertsdifferenz.
    """
    # Prüfung, ob h positiv ist
    if(h<=0):
        raise ValueError('h muss positiv sein')
    
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

def  Mitteldifferenz(f, x, h):
    """
    :param f: Funktion, deren Ableitung mithilfe der Mitteldifferenz approximiert werden soll.
    :param x: Vektor als np.array, an dem Ableitung approximiert werden soll.
    :param h: Schrittweite, aufgrund deren der Differenzenquotient gebildet wird.
    :raise ValueError: Wenn `h <= 0`.
    :return: Approximierung von `f'(x)` mithilfe der Mitteldifferenz.
    """
    # Prüfung, ob h positiv ist
    if(h<=0):
        raise ValueError('h muss positiv sein')
    
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

def schrittGradientenverfahren_festeSchrittweite(nablaF, x, schritt):
    """
    :param nableF: Ableitung der Funktion, die minimiert werden soll.
    :param x: Startwert für das Gradientenverfahren.
    :param schritt: Schrittweite, wie weit der Schritt des Gradientenverfahrens in die berechnete Richtung geht.
    :return: berechnete neue Stelle `x`.
    """

    abl_real = nablaF(x).real
    return x-schritt*abl_real

# das beruht auf echter Trapezregel, wo man Mittelwert bildet und Differenz der Stellen
def quadratureContourIntegralCircleTrapez(f, gamma:complex, r:float, m:int, s) -> complex:
    # Berechne Stützstellen
    zs = np.array([gamma+r*np.exp(2*np.pi*1j/m*(k+1/2)) for k in range(m+1)])
    return sum((f(zs[i+1], s)+f(zs[i], s))/2*(zs[i+1]-zs[i]) for i in range(m))

# das ist eher die Mittelpunktsregel, wo man Funktion nur an einer Stelle auswerten muss, Differenz wurde explizit berechnet und rausgezogen
def quadratureContourIntegralCircleMittelpunkt(f, gamma:complex, r:float, m:int, s) -> complex:
    # Berechne Stützstellen
    zs = np.array([gamma+r*np.exp(2*np.pi*1j/m*(k+1/2)) for k in range(m)])
    return sum(f(zs[i], s)*np.exp(2*np.pi*1j*i/m) for i in range(m))*r*(np.exp(2*np.pi*1j/m)-1)

# minimiert die gewichtete Zaehlung der Eigenwerte, berechnet tatsächliche gewichtete Eigenwert-Zaehlung, gibt alles aus
def EigenwerteMinimierenAufIntervall(M:np.ndarray, K:np.ndarray, startpunkt:np.ndarray, lambda_a, lambda_b, anzahlStuetzstellen:int, schrittweiteGradVerf:float, fehlertoleranz:float) -> np.ndarray:
    """
    :param M: Massematrix.
    :param K: Steifigkeitsmatrix.
    :param lambda_a: untere Grenze des Intervalls.
    :param lambda_b: obere Grenze des Intervalls.
    :anzahlStuetzstellen: Anzahl der Stützstellen in Quadraturformel.
    :param schrittweiteGradVerf: Schrittweite, wie weit ein Schritt des Gradientenverfahrens in die berechnete Richtung geht.
    :param fehlertoleranz: Grenze, wenn J(s) drunter fällt, dann wird Minimierungsverfahren beendet.
    :return: Vektor, 1. Spalte: berechneten Werte s, 2. Spalte: gewichtete Zählung (genau), 3. Spalte: gewichtete Zählung (approximiert).
    """

    z0Tilde = (lambda_a+lambda_b)/2
    rTilde = (lambda_b-lambda_a)/2
    len_s = len(startpunkt)

    # Gewichtungsfunktion g(z)
    def g(z: complex) -> complex:
        return -(z-((1+alpha)*lambda_a-alpha*lambda_b))*(z-((1+alpha)*lambda_b-alpha*lambda_a))
    
    # berechnet die Eigenwerte mittels funktion np.linalg.eigvals, siehe Definition mu
    def gewichteteEWZaehlung_genau(s)->float:
        # ist \1_{[\lambda_a,\lambda_b]}(z)
        def h(z:float)->float:
            if(z < lambda_b and z>lambda_a):
                return 1.
            return 0.
        
        eigvals = np.linalg.eigvals(np.linalg.inv(M(s)).dot(K(s)))
        return sum(h(i)*g(i) for i in eigvals)

    def J_Stern(s)->float:
        # zur besseren Lesbarkeit wird B = (zM-K)^{-1} M ausgelagert
        def B(z:complex, s:np.ndarray)-> np.ndarray[complex]:
            D = np.linalg.inv(z*M(s)-K(s))
            return g(z)*np.trace(D.dot(M(s)))
        
        return quadratureContourIntegralCircleTrapez(B, z0Tilde, rTilde, anzahlStuetzstellen, s)/2/np.pi/1j
    
    def nablaJ_Stern(s)->float:
        def nablaB(z:complex, s:np.ndarray)->np.ndarray[complex]:
            D = np.linalg.inv(z*M(s)-K(s))
            dMds = Vorwaertsdifferenz(M, s, schrittweiteDiffverfahren)
            dKds = Vorwaertsdifferenz(K, s, schrittweiteDiffverfahren)

            return g(z)*(np.trace(D.dot(dMds))-np.trace(D.dot(z*dMds-dKds).dot(D)))
        return quadratureContourIntegralCircleTrapez(nablaB, z0Tilde, rTilde, anzahlStuetzstellen, s)/2/np.pi/1j

################## hier wird in jedem Schritt mehrmals M(s) und K(s) berechnet ##################

    # result wird eine 2d-Matrix, result[0:j-1,i] ist Wert, der in i-tem Schritt berechnet wurde
    # result[-2,i] genaue gewichtete Zaehlung der Eigenwerte mit Wert aus i-tem Schritt
    # result[-1,i] approximierte gewichtete Zaehlung der Eigenwerte mit Wert aus i-tem Schritt

    result = np.expand_dims(np.append(startpunkt, [gewichteteEWZaehlung_genau(startpunkt), J_Stern(startpunkt)]), 1)

    while result[-1,-1]>=fehlertoleranz:
        x_neu = schrittGradientenverfahren_festeSchrittweite(nablaJ_Stern, np.array(result[0:len_s,-1]), schrittweiteGradVerf)

        result = np.append(result, np.expand_dims(np.append(x_neu, [gewichteteEWZaehlung_genau(x_neu), J_Stern(x_neu)]), 1), axis=1)

        if(np.size(result,1)%100 == 0):
            print(result[-2,-1])
            if(np.size(result,1) == 500):
                break


        

    return result