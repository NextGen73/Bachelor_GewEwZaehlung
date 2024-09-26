import Funktionen
from math import pi

zweiPiI = 2j*pi
sArray = [0.9, 0.99, 0.999, 0.9999]

def f(z, s) -> complex:
    return 1/(z-s)
print("\n\ndie Integrationskurve ist immer r(t)={exp(it): t \\in [0, 2*pi]}, sie wird immer in 100 Teilintervalle unterteilt, welche approximiert werden")
for s in sArray:
    print("integriere f(z)=1/(z-",s,"):\n")

    print("einfache Trapezformel: ", Funktionen.quadratureContourIntegralCircleTrapez(f, s, 100)/zweiPiI)
    print("Mittelpunktformel: ", Funktionen.quadratureContourIntegralCircleMittelpunkt(f, s, 100)/zweiPiI)
    print("verschobene Trapezformel: ", Funktionen.quadratureContourIntegralCircleTrapezNeu(f, s, 100)/zweiPiI)
    print("2 Punktformel nach Gauss: ", Funktionen.quadratureContourIntegralCircleGaussZwei(f, s, 100)/zweiPiI)
    print("")

print("wie man sehen kann, wächst die einfache Trapezformel immer weiter, je näher sich die Polstelle der Integrationskurve nähert")
print("die Realteile der anderen Quadraturformeln gehen hingegen gegen 0.5, allerdings ist der Imaginärteil bei der verschobenen Trapezformel viel größer als bei allen anderen Quadraturen")