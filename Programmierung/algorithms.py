import numpy as np

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

def gradientenverfahren_festeSchrittweite(nablaF, x0, schritt, epsilon):
    """
    :param nableF: Ableitung der Funktion, die minimiert werden soll.
    :param x0: Startwert für das Gradientenverfahren.
    :param schritt: Schrittweite, wie weit ein Schritt des Gradientenverfahrens in die berechnete Richtung geht.
    :param epsilon: Grenze, die den Abbruch des Verfahrens bewirkt, falls `||f'(x_k)|| < epsilon`.
    :return: Liste der `x_k`, die das Verfahren berechnet hat.
    """

    xArray = np.array([x0])
    i=0
    x_alt = x0
    while np.norm(nablaF(xArray[i]))>=epsilon:
        x_neu = xArray[i]-schritt*nablaF(xArray[i])

        np.append(xArray, x_neu)
        x_alt = x_neu
        i+=1

        if(i>=10000):
            break

    return xArray