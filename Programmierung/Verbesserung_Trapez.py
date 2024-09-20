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
    print("An den folgenden 2 Beispielen kann man sehen, dass eine genauere Quadratur hier keinen richtigen Unterschied machen muss.")
    print("Beachte, wie Eigenwert 2 sich beide Male langsam der Intervallgrenze nähert, es aber nicht schafft sie zu überqueren.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05)
    # Verwendung von mehr Teilintervallen
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05)

    print("Erhöhe nun die Schrittweite des Gradientenverfahrens, um eine schnellere Konvergenz zu erwarten.")
    print("Man siehe zudem, dass eine höhere Genauigkeit der Quadraturformel hier zu einem schnelleren Ergebnis führt.\n")
    # Verfahren mit Standardwerten, aber groesserer Schrittweite des Gradientenverfahrens
    minimierenPlottenUndEckdatenAnzeigen(100, 0.5)
    # wieder mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.5)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    print("Mit diesen zwei Durchläufen wird gezeigt, dass mehr Teilintervalle sogar schlecht sein können.\n")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05)
    # mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05)