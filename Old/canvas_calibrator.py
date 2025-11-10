
# canvas_calibrator.py
# Clicca prima l'angolo in alto-sinistra del foglio in Samsung Notes,
# poi l'angolo in basso-destra. Il programma calcola width/height in pixel.
import time
import pyautogui as pag

pag.FAILSAFE = True

print("Tra 4 secondi, passa a Samsung Notes.")
time.sleep(4)
print("1) Posiziona il mouse sull'ANGOLO ALTO-SINISTRA del foglio e premi INVIO qui nel terminale.")
input()
x1, y1 = pag.position()
print(f"Top-Left: ({x1}, {y1})")

print("2) Posiziona il mouse sull'ANGOLO BASSO-DESTRA del foglio e premi INVIO qui nel terminale.")
input()
x2, y2 = pag.position()
print(f"Bottom-Right: ({x2}, {y2})")

w = x2 - x1
h = y2 - y1
print(f"\nCanvas size rilevato: width={w} px, height={h} px")
print(f"Origine (da usare con l'autowriter): ({x1}, {y1})")
