from Funktionen import minimierenPlottenUndEckdatenAnzeigen, systemAuswaehlen, init, quadratureContourIntegralCircleTrapezNeu

# Dimension der Masse- und Steifigkeitsmatrix, damit auch Anzahl an Massen in System und Anzahl Freiheitsgrade
n = 8
# j gibt für System 1 an, ab welchem Index die Masse fest ist
# für System 2 gibt es die Länge des Vektors s an
j = int(n/2)

# durch initSystem wird ferner die Anzahl der Teilintervalle immer auf den Standard (m=100) zurueckgesetzt
if(__name__=='__main__'):

    # beachte, dass man immer init() und systemAuswaehlen() aufrufen muss
    # ohne diese Aufrufe koennen keine Berechnungen durchgeführt werden
    
    init(quadratureContourIntegralCircleTrapezNeu, n, j, maxIter=500)
    print("Verschiebe in diesen Durchläufen die Stützstellen der Trapezregel so, dass keine Stützstelle bei lambda_a oder lambda_b liegt.")
    print("Allein das reicht aus, damit die Minimierung beschleunigt wird.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05)
    # Verwendung von mehr Teilintervallen
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05)

    print("Erhöhe nun die Schrittweite des Gradientenverfahrens, um eine schnellere Konvergenz zu erwarten.")
    print("Man sieht auch, dass die gewichtete Eigenwertzählung jetzt viel genauer approximiert wird (im Vergleich zu 'erste Implementierung.py'), falls ein Eigenwert nahe des Intervallrandes liegt.\n")
    # Verfahren mit Standardwerten, aber groesserer Schrittweite des Gradientenverfahrens
    minimierenPlottenUndEckdatenAnzeigen(100, 0.5)
    # wieder mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.5)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    print("Auch hier bewirkt die leicht veränderte Quadraturformel eine verbesserte Performance.")
    print("Diese Performance kann zwar nicht direkt im Durchlauf mit m=100 gesehen werden, aber die Sprünge in der Verteilung der Eigenwerte wurden viel kleiner, was auf eine bessere Approximation schließen lässt.\n")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05)
    # mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05)