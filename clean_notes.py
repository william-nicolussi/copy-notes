import cv2
import numpy as np
from pathlib import Path

def clean_handwritten_notes(input_path: str, output_path: str):
    """
    Rimuove la griglia, appiattisce lo sfondo a bianco puro, e rende tutto il testo nero.
    Pensato per appunti scritti a mano su carta quadrettata con inchiostri colorati.
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {input_path}")

    # 1) Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Stima dello sfondo (median blur "forte") e normalizzazione per sopprimere la griglia
    bg = cv2.medianBlur(gray, 21)
    bg = np.clip(bg, 1, 255)
    norm = cv2.divide(gray, bg, scale=255)

    # 3) Binarizzazione (Otsu). Sfondo bianco, inchiostro nero
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Garantisce testo nero e sfondo bianco (inverti se i bordi risultano scuri)
    border = np.concatenate([bw[0,:], bw[-1,:], bw[:,0], bw[:,-1]])
    if border.mean() < 127:
        bw = cv2.bitwise_not(bw)

    # 4) Rimozione puntinature/avanzi di griglia: elimina componenti troppo piccole
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(255 - bw, connectivity=8)
    cleaned = bw.copy()
    min_area = 18  # aumenta se restano puntini di griglia; diminuisci se mangia dettagli
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 255

    # 5) Piccola apertura morfologica per pulire senza assottigliare troppo
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cleaned)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pulisci appunti: rimuovi griglia e rendi il testo nero su bianco.")
    ap.add_argument("input", help="Percorso immagine in ingresso (jpg/png/pdf convertito in immagine).")
    ap.add_argument("output", help="Percorso immagine in uscita (png/jpg).")
    args = ap.parse_args()
    clean_handwritten_notes(args.input, args.output)
