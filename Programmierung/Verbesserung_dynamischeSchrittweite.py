from algorithms import minimierenPlottenUndEckdatenAnzeigen, systemAuswaehlen, init
import algorithms

# Dimension der Masse- und Steifigkeitsmatrix, damit auch Anzahl an Massen in System und Anzahl Freiheitsgrade
n = 8
# j gibt für System 1 an, ab welchem Index die Masse fest ist
# für System 2 gibt es die Länge des Vektors s an
j = int(n/2)

# durch initSystem wird ferner die Anzahl der Teilintervalle immer auf den Standard (m=100) zurueckgesetzt
if(__name__=='__main__'):

    # beachte, dass man immer init() und systemAuswaehlen() aufrufen muss
    # ohne diese Aufrufe koennen keine Berechnungen durchgeführt werden
    
    init(algorithms.quadratureContourIntegralCircleMittelpunkt, n, j)
    print("An den folgenden 2 Beispielen kann man sehen, dass eine genauere Quadratur hier keinen richtigen Unterschied machen muss.")
    print("Beachte, wie Eigenwert 2 sich beide Male langsam der Intervallgrenze nähert, es aber nicht schafft sie zu überqueren.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)

    minimierenPlottenUndEckdatenAnzeigen(100, dynamischSchritt=True)
    # Verwendung von mehr Teilintervallen
    minimierenPlottenUndEckdatenAnzeigen(150, dynamischSchritt=True)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    print("Mit diesen zwei Durchläufen wird gezeigt, dass mehr Teilintervalle sogar schlecht sein können.\n")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)

    minimierenPlottenUndEckdatenAnzeigen(100, dynamischSchritt=True)
    # mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, dynamischSchritt=True)

    print("Nehme nun eine größere Schrittweite")
    minimierenPlottenUndEckdatenAnzeigen(100, 0.5, True)
    minimierenPlottenUndEckdatenAnzeigen(150, 0.5, True)