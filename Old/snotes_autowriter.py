# snotes_autowriter.py
# Bot che "riscrive" un'immagine (pagina A4 ecc.) dentro Samsung Notes simulando tratti di penna.
# Include: rimozione griglia, soglia adattiva (Otsu/Sauvola), upscaling anti-alias, opzione --invert,
# anteprima dei file intermedi (clean_no_grid.png, clean_bw.png).
#
# Esempi:
# 1) Solo anteprima (verifica maschera): 
#    python .\snotes_autowriter.py --image "Screenshot.png" --width 650 --height 914 --preview_only --th_mode otsu --scale_up 2 --row_skip 1 --stroke_step 2 --anti_alias
# 2) Disegno reale:
#    python .\snotes_autowriter.py --image "Screenshot.png" --width 650 --height 914 --th_mode otsu --scale_up 2 --row_skip 1 --stroke_step 2 --anti_alias
#
import argparse
import time
import pyautogui as pag
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

pag.FAILSAFE = True  # sposta il mouse nell'angolo alto-sx dello schermo per interrompere

def resize_to_canvas(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    # Ridimensiona al canvas finale con filtro di alta qualità
    return im.resize((target_w, target_h), Image.LANCZOS)

def rgb_to_sv(arr: np.ndarray):
    # Restituisce Saturation e Value (HSV) per mascherare lo sfondo
    cmax = np.max(arr, axis=-1)
    cmin = np.min(arr, axis=-1)
    delta = cmax - cmin + 1e-6
    s = delta / (cmax + 1e-6)
    v = cmax
    return s, v

def remove_grid(im: Image.Image, s_thr: float, v_thr: float, thicken: int) -> Image.Image:
    # Elimina griglia/sfondo: tiene pixel colorati (S alta) o scuri (V basso)
    arr = np.asarray(im).astype(np.float32) / 255.0
    s, v = rgb_to_sv(arr)
    mask_keep = (s > s_thr) | (v < v_thr)
    if thicken and thicken > 1:
        mask_keep = maximum_filter(mask_keep.astype(np.uint8), size=thicken).astype(bool)
    clean = np.ones_like(arr)
    clean[mask_keep] = arr[mask_keep]
    clean = np.clip(clean, 0, 1)
    return Image.fromarray((clean * 255).astype(np.uint8))

def otsu_threshold(gray: np.ndarray) -> float:
    # gray float [0..1]
    hist, bin_edges = np.histogram(gray, bins=256, range=(0, 1))
    hist = hist.astype(np.float64)
    prob = hist / np.maximum(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.linspace(0, 1, 256))
    mu_t = mu[-1]
    denom = (omega * (1 - omega))
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b))
    return float(bin_edges[idx])

def sauvola_threshold(gray: np.ndarray, window: int = 25, k: float = 0.2, R: float = 0.5) -> np.ndarray:
    # Soglia locale (Sauvola) su gray float [0..1]
    mean = uniform_filter(gray, size=window, mode='reflect')
    mean_sq = uniform_filter(gray * gray, size=window, mode='reflect')
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    th = mean * (1 + k * (std / (R + 1e-6) - 1))
    return th

def binarize_adaptive(im: Image.Image, mode: str, fixed: float, sauvola_w: int, sauvola_k: float) -> Image.Image:
    # Ritorna BW con 0=nero (TRACCIA) e 255=bianco (SFONDO)
    arr = np.asarray(im).astype(np.float32) / 255.0
    gray = np.dot(arr, [0.299, 0.587, 0.114])
    if mode == "fixed":
        th = fixed
        bw = np.where(gray < th, 0, 255).astype(np.uint8)
    elif mode == "otsu":
        th = otsu_threshold(gray)
        bw = np.where(gray < th, 0, 255).astype(np.uint8)
    elif mode == "sauvola":
        th_map = sauvola_threshold(gray, window=sauvola_w, k=sauvola_k, R=0.5)
        bw = np.where(gray < th_map, 0, 255).astype(np.uint8)
    else:
        raise ValueError("th_mode non valido (fixed/otsu/sauvola)")
    return Image.fromarray(bw, mode="L")

def collect_strokes_rowwise(bw_img: Image.Image, stroke_step: int = 3, row_skip: int = 2):
    # Estrae segmenti orizzontali neri (0) contigui ogni row_skip righe
    w, h = bw_img.size
    px = bw_img.load()
    strokes = []
    for y in range(0, h, row_skip):
        x = 0
        while x < w:
            while x < w and px[x, y] != 0:  # salta bianchi (255)
                x += 1
            if x >= w:
                break
            x0 = x
            while x < w and px[x, y] == 0:
                x += 1
            x1 = x - 1
            cur = x0
            while cur <= x1:
                nxt = min(cur + stroke_step, x1)
                strokes.append([(cur, y), (nxt, y)])
                cur = nxt + 1
    return strokes

def draw_strokes_on_canvas(strokes, origin_x: int, origin_y: int, speed: float = 0.0, pen_down_pause: float = 0.002):
    # Simula movimenti del mouse per disegnare micro-tratti
    for (x0, y0), (x1, y1) in strokes:
        pag.moveTo(origin_x + x0, origin_y + y0, duration=speed)
        pag.mouseDown()
        pag.moveTo(origin_x + x1, origin_y + y1, duration=speed)
        if pen_down_pause > 0:
            time.sleep(pen_down_pause)
        pag.mouseUp()

def main():
    ap = argparse.ArgumentParser(description="Autowriter Samsung Notes con qualità migliorata e anti-griglia.")
    ap.add_argument("--image", required=True, help="Immagine della pagina (PNG/JPG)")
    ap.add_argument("--width", type=int, required=True, help="Larghezza canvas px")
    ap.add_argument("--height", type=int, required=True, help="Altezza canvas px")
    ap.add_argument("--antigrid_s", type=float, default=0.22, help="Soglia Saturation per tenere inchiostro")
    ap.add_argument("--antigrid_v", type=float, default=0.35, help="Soglia Value per tenere neri/scuri")
    ap.add_argument("--dilate", type=int, default=2, help="Ispessimento max-filter dopo anti-griglia")
    ap.add_argument("--th_mode", choices=["fixed", "otsu", "sauvola"], default="otsu", help="Metodo soglia")
    ap.add_argument("--bw_thresh", type=float, default=0.65, help="Soglia fissa se th_mode=fixed (0..1)")
    ap.add_argument("--sauvola_w", type=int, default=23, help="Finestra Sauvola")
    ap.add_argument("--sauvola_k", type=float, default=0.18, help="Parametro k Sauvola")
    ap.add_argument("--scale_up", type=int, default=2, help="Upscale pre-process (1=off)")
    ap.add_argument("--anti_alias", action="store_true", help="Leggera sfocatura prima della soglia (bordo più pieno)")
    ap.add_argument("--stroke_step", type=int, default=2, help="Lunghezza micro-tratto (px)")
    ap.add_argument("--row_skip", type=int, default=1, help="Densità righe (1=max qualità)")
    ap.add_argument("--speed", type=float, default=0.0, help="Durata moveTo per segmento (0=istantaneo)")
    ap.add_argument("--pen_pause", type=float, default=0.002, help="Pausa penna giù (s)")
    ap.add_argument("--invert", action="store_true", help="Inverti B/N dopo la binarizzazione (se disegna lo sfondo)")
    ap.add_argument("--preview_only", action="store_true", help="Non disegna: salva solo clean_no_grid.png e clean_bw.png")
    args = ap.parse_args()

    # Carica immagine
    im = Image.open(args.image).convert("RGB")
    im = resize_to_canvas(im, args.width, args.height)

    # Upscaling per preservare bordi sottili
    if args.scale_up and args.scale_up > 1:
        im = im.resize((im.width * args.scale_up, im.height * args.scale_up), Image.LANCZOS)

    # Rimozione griglia/sfondo
    clean = remove_grid(im, s_thr=args.antigrid_s, v_thr=args.antigrid_v, thicken=args.dilate)

    # Anti-alias opzionale
    if args.anti_alias:
        clean = clean.filter(ImageFilter.GaussianBlur(radius=0.7))

    # Binarizzazione di qualità (0=nero tratti, 255=bianco sfondo)
    bw = binarize_adaptive(clean, mode=args.th_mode, fixed=args.bw_thresh, sauvola_w=args.sauvola_w, sauvola_k=args.sauvola_k)

    # Riporta alla misura finale se era in upscaling
    if args.scale_up and args.scale_up > 1:
        clean = clean.resize((args.width, args.height), Image.LANCZOS)
        bw = bw.resize((args.width, args.height), Image.NEAREST)

    # Inversione opzionale
    if args.invert:
        bw = ImageOps.invert(bw)

    # Salva anteprime utili
    clean.save("clean_no_grid.png")
    bw.save("clean_bw.png")
    print("[OK] Salvati: clean_no_grid.png (colori), clean_bw.png (maschera tratti: NERO=verrà disegnato).")

    if args.preview_only:
        print("Preview only attivo: verifica clean_bw.png. Se ok, rilancia senza --preview_only.")
        return

    # Allineamento origine canvas
    print("Tra 5 secondi passa a Samsung Notes.")
    time.sleep(5)
    print("Posiziona il mouse sull'ANGOLO ALTO-SINISTRA del foglio e premi INVIO qui nel terminale.")
    input()
    origin_x, origin_y = pag.position()
    print(f"Origine set a: ({origin_x}, {origin_y}). Avvio in 3 secondi…")
    time.sleep(3)

    # Genera tratti e disegna
    strokes = collect_strokes_rowwise(bw, stroke_step=args.stroke_step, row_skip=args.row_skip)
    print(f"[INFO] micro-tratti da disegnare: {len(strokes)}")
    t0 = time.time()
    draw_strokes_on_canvas(strokes, origin_x, origin_y, speed=args.speed, pen_down_pause=args.pen_pause)
    print(f"[FINE] Tempo: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
