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
    print("Da die Quadratur nicht mehr so empfindlich gegenüber Eigenwerten nahe der Integrationskurve ist, bewirkt eine Quadraturformel mit mehr Teilintervallen bei System 1 kaum einen Unterschied.")
    print("Alle folgenden Durchläufe werden mit der Zweipunkt-Formel von Gauss durchgeführt.")
    print("Die ersten zwei sind wieder System 1 mit kleiner Schrittweite des Gradientenverfahrens, beim zweiten Plot wurde die Integrationskurve in mehr Teilintervalle aufgeteilt.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05)
    # Verwendung von mehr Teilintervallen
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05)

    print("Erhöhe nun die Schrittweite des Gradientenverfahrens, um eine schnellere Konvergenz zu erwarten.")
    print("Wie schon oben ist der Unterschied zwischen bei verschiedener Anzahl an Teilintervallen minimal.")
    print("Die Durchläufe benötigen nur ein Zehntel der Schritte, da die Schrittweite hier zehnmal größer als oben ist.\n")
    # Verfahren mit Standardwerten, aber groesserer Schrittweite des Gradientenverfahrens
    minimierenPlottenUndEckdatenAnzeigen(100, 0.5)
    # wieder mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.5)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    print("Obwohl die Eigenwerte zwischenzeitlich sehr groß werden, werden die gewichteten Eigenwertzählungen in diesen Durchläufen zügig minimiert.")
    print("Man beachte, dass die Erhöhung der Anzahl an Teilintervallen hier zur Folge hat, dass die Anzahl an Schritten halbiert wird.")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)

    minimierenPlottenUndEckdatenAnzeigen(100, 0.05)
    # mehr Teilintervalle verwenden
    minimierenPlottenUndEckdatenAnzeigen(150, 0.05)