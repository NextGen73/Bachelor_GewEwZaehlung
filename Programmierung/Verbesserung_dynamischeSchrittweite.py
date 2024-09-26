from Funktionen import minimierenPlottenUndEckdatenAnzeigen, systemAuswaehlen, init, quadratureContourIntegralCircleGaussZwei

# Dimension der Masse- und Steifigkeitsmatrix, damit auch Anzahl an Massen in System und Anzahl Freiheitsgrade
n = 8
# j gibt für System 1 an, ab welchem Index die Masse fest ist
# für System 2 gibt es die Länge des Vektors s an
j = int(n/2)

# durch initSystem wird ferner die Anzahl der Teilintervalle immer auf den Standard (m=100) zurueckgesetzt
if(__name__=='__main__'):

    # beachte, dass man immer init() und systemAuswaehlen() aufrufen muss
    # ohne diese Aufrufe koennen keine Berechnungen durchgeführt werden
    
    init(quadratureContourIntegralCircleGaussZwei, n, j)
    print("An den folgenden Beispiel kann man sehen, dass durch die variable Schrittweite hier erreicht wird, dass das Verfahren nach einem Schritt fertig ist.")
    print("Das ist aber nicht verwunderlich, da der Parameter eindimensional ist.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)
    minimierenPlottenUndEckdatenAnzeigen(100, dynamischSchritt=True)
    minimierenPlottenUndEckdatenAnzeigen(150, dynamischSchritt=True)

    # verwende nun wieder eine größere Reichweite
    print("Durch eine größere Schrittweite wird das Verfahren beschleunigt, da nun nicht mehr so viele Teilschritte benötigt werden.")
    print("Die Anzahl an kompletten Schritten bleibt daher 1, aber die Zahl an Unterschritten wird geringer.\n")
    minimierenPlottenUndEckdatenAnzeigen(100, 0.5, dynamischSchritt=True)
    minimierenPlottenUndEckdatenAnzeigen(150, 0.5, dynamischSchritt=True)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    print("Hier wird durch die variable Schrittweite zwar mehr Zeit benötigt, aber die Anzahl an benötigten Schritten verringert sich stark für m=100.")
    print("Bei m=150 wird aber viel mehr Zeit benötigt, da die Schritte 2 bis 173 hier die Verteilung der Eigenwerte kaum verändern.")
    print("Dies ist insbesondere schlecht, da durch die dynamische Schrittweite sichergestellt werden sollte, dass solche langsamen Veränderungen nicht mehr auftretetn.\n")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)
    minimierenPlottenUndEckdatenAnzeigen(100, dynamischSchritt=True)
    minimierenPlottenUndEckdatenAnzeigen(150, dynamischSchritt=True)

    print("Da bei jedem einzelnen Durchlauf mehr Zeit benötigt wird als in Verbesserung_Gauss2.py oder Verbesserung_Mittelpunkt.py ist eine variable Schrittweite nicht besser als eine feste.")
    print("Dies liegt aber hauptsächlich an dem Umstand, dass die optimale Schrittweite nicht berechnet werden kann, sondern die Gerade an vielen Punkten ausgewertet werden muss.")