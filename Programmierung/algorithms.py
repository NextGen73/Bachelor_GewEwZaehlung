import numpy as np

def  Vorwaertsdifferenz(f, x, h):
    """
    :param f: Funktion, deren Ableitung mithilfe der Vorwaertsdifferenz approximiert werden soll.
    :param x: Wert, an dem Ableitung approximiert werden soll.
    :param h: Schrittweite, aufgrund deren der Differenzenquotient gebildet wird.
    :raise ValueError: Wenn `h <= 0`.
    :return: Approximierung von `f'(x)` mithilfe der Vorwaertsdifferenz.
    """
    if(h<=0):
        raise ValueError('h muss positiv sein')
    
    return (f(x+h)-f(x))/h

def  Rueckwaertsdifferenz(f, x, h):
    """
    :param f: Funktion, deren Ableitung mithilfe der Rueckwaertsdifferenz approximiert werden soll.
    :param x: Wert, an dem Ableitung approximiert werden soll.
    :param h: Schrittweite, aufgrund deren der Differenzenquotient gebildet wird.
    :raise ValueError: Wenn `h <= 0`.
    :return: Approximierung von `f'(x)` mithilfe der Rueckwaertsdifferenz.
    """
    if(h<=0):
        raise ValueError()
    
    return (f(x)-f(x-h))/h

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
