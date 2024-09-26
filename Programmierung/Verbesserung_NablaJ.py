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
    
    init(quadratureContourIntegralCircleGaussZwei, n, j, maxIter=100000)
    print("Approximiere nun nabla J durch ein Differenzenverfahren. Damit spart man sich in jedem Schritt die Ausführung der Quadraturformel mit m Teilintervallen")
    print("Man erhofft sich dadurch eine starke Beschleunigung der Minimierung.")
    print("Die Anzahl an Teilintervallen wird hier nur noch benötigt, um die gewichtete Eigenwertzählung in jedem Schritt des Gradientenverfahrens zu approximieren.")
    print("Unterscheide daher nicht mehr nach Anzahl Teilintervalle m und setze m auf 1, um den Mehraufwand so gering wie möglich zu halten.\n")
    # System 1 mit Standardwerten initialisieren
    systemAuswaehlen(1)

    minimierenPlottenUndEckdatenAnzeigen(1, 0.05, approxNablaJ=True)

    print("Erhöhe nun die Schrittweite des Gradientenverfahrens, um eine schnellere Konvergenz zu erwarten.")
    print("Hier ist der Effekt nicht gut sichtbar, da auch schon der erste Durchlauf innerhalb von einem Zehntel einer Sekunde beendet war.\n")
    # Verfahren mit Standardwerten, aber groesserer Schrittweite des Gradientenverfahrens
    minimierenPlottenUndEckdatenAnzeigen(1, 0.5, approxNablaJ=True)

    print("Betrachte nun das zweite System mit kleiner Schrittweite.")
    print("Dieses Verfahren ist in dieser Form zu ungenau, als dass es die gewichtete Eigenwertzählung bei System 2 sinnvoll minimieren zu können.")
    print("Durch etwaige Verbesserungen könnte aber dieser Fehler aufgehoben werden.\n")
    # System 2 mit Standardwerten
    systemAuswaehlen(2)

    minimierenPlottenUndEckdatenAnzeigen(1, 0.05, approxNablaJ=True)
