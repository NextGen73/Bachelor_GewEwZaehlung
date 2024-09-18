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
    
    init(algorithms.quadratureContourIntegralCircleGaussZwei, n, j)
    print("An dem folgenden Beispiel kann man sehen, dass durch die variable Schrittweite hier erreicht wird, dass das Verfahren nach einem Schritt fertig ist.")
    print("Das ist aber nicht verwunderlich, da der Parameter eindimensional ist.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)
    minimierenPlottenUndEckdatenAnzeigen(100, dynamischSchritt=True)
    minimierenPlottenUndEckdatenAnzeigen(150, dynamischSchritt=True)

    # verwende nun wieder eine größere Reichweite
    minimierenPlottenUndEckdatenAnzeigen(100, 0.5, dynamischSchritt=True)
    minimierenPlottenUndEckdatenAnzeigen(150, 0.5, dynamischSchritt=True)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    print("Hier wird durch die variable Schrittweite zwar mehr Zeit benötigt, aber dafür bleiben die Eigenwerte bis zum Schritt 120 sehr oft fast konstant.\n")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)
    minimierenPlottenUndEckdatenAnzeigen(100, dynamischSchritt=True)
    minimierenPlottenUndEckdatenAnzeigen(150, dynamischSchritt=True)