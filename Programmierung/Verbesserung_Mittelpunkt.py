from Funktionen import minimierenPlottenUndEckdatenAnzeigen, systemAuswaehlen, init, quadratureContourIntegralCircleMittelpunkt

# Dimension der Masse- und Steifigkeitsmatrix, damit auch Anzahl an Massen in System und Anzahl Freiheitsgrade
n = 8
# j gibt für System 1 an, ab welchem Index die Masse fest ist
# für System 2 gibt es die Länge des Vektors s an
j = int(n/2)

# durch initSystem wird ferner die Anzahl der Teilintervalle immer auf den Standard (m=100) zurueckgesetzt
if(__name__=='__main__'):

    # beachte, dass man immer init() und systemAuswaehlen() aufrufen muss
    # ohne diese Aufrufe koennen keine Berechnungen durchgeführt werden
    
    init(quadratureContourIntegralCircleMittelpunkt, n, j)
    print("Da die Quadratur nicht mehr so empfindlich gegenüber Eigenwerten nahe der Integrationskurve ist, macht eine Quadraturformel mit mehr Teilintervallen kaum einen Unterschied mehr.")
    print("Alle folgenden Durchläufe werden mit der Mittelpunktformel durchgeführt.")
    print("Die ersten zwei sind wieder System 1 mit kleiner Schrittweite des Gradientenverfahrens, beim zweiten Plot wurde die Integrationskurve in mehr Teilintervalle aufgeteilt.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05)
    # Verwendung von mehr Teilintervallen
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05)

    print("Erhöhe nun die Schrittweite des Gradientenverfahrens, um eine schnellere Konvergenz zu erwarten.")
    print("Wie bei allen Programmen, welche die Eigenwertzählung bei System 1 zuverlässig minimieren können,")
    print("verringert sich die benötigte Zeit, da durch die erhöhte Schrittweite weniger Stellen berechnet werden müssen.\n")
    # Verfahren mit Standardwerten, aber groesserer Schrittweite des Gradientenverfahrens
    minimierenPlottenUndEckdatenAnzeigen(100, 0.5)
    # wieder mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.5)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    print("Man sieht hier sehr gut, dass die approximierte gewichtete Eigenwertzählung ungenau wird, falls ein Eigenwert sich einer Intervallgrenze nähern sollte.")
    print("An genau diesen Stellen findet man in den Verläufen der Eigenwerte auch einen Sprung.\n")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05)
    # mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05)

    print("Im Vergleich zu 'Verbesserung_Gauss2.py' sind die Zeiten bei System 1 geringer. Bei einem mehrdimensionalen Design-Parameter, wie bei System 2, ist diese Variante aber nicht so performant.")