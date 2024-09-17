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
    print("Da die Quadratur nicht mehr so empfindlich gegenüber Eigenwerten nahe der Integrationskurve ist, macht eine Quadraturformel mit mehr Teilintervallen kaum einen Unterschied mehr.")
    print("Alle folgenden Durchläufe werden mit der verschobenen Trapezformel durchgeführt.")
    print("Die ersten zwei sind wieder System 1 mit kleiner Schrittweite des Gradientenverfahrens, beim zweiten Plot wurde die Integrationskurve in mehr Teilintervalle aufgeteilt.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05, approxNablaJ=True)
    # Verwendung von mehr Teilintervallen
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05, approxNablaJ=True)

    print("Erhöhe nun die Schrittweite des Gradientenverfahrens, um eine schnellere Konvergenz zu erwarten.")
    # Verfahren mit Standardwerten, aber groesserer Schrittweite des Gradientenverfahrens
    minimierenPlottenUndEckdatenAnzeigen(100, 0.5, approxNablaJ=True)
    # wieder mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.5, approxNablaJ=True)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05, approxNablaJ=True)
    # mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05, approxNablaJ=True)