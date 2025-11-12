# core/pdf_io.py
from pathlib import Path

def run_fase1(base_dir: Path, pdf_path: Path, params: dict, logger=print):
    """
    PDF → PNG in original/ e cleaned/ usando la stessa procedura del tuo .pyw:
    - render con PyMuPDF
    - conversione BGRA→BGR
    - anti-grid 'keep ink' su cleaned (identico a hsv_antigrid_keep_ink_bgr del .pyw)
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        logger(f"[F1][ERR] PyMuPDF non disponibile: {e}")
        logger("[F1] Installa con: pip install pymupdf")
        return

    try:
        import cv2, numpy as np
    except Exception as e:
        logger(f"[F1][ERR] OpenCV non disponibile: {e}")
        logger("[F1] Installa con: pip install opencv-python")
        return

    if not pdf_path or not pdf_path.exists():
        logger("[F1][ERR] PDF non trovato o non selezionato.")
        return

    dpi       = int(params.get("dpi", 600))
    anti_grid = bool(params.get("anti_grid", True))
    s_thr     = float(params.get("s_thr", 0.26))
    v_thr     = float(params.get("v_thr", 0.65))
    thicken   = int(params.get("thicken", 2))

    dir_original = base_dir / "original"
    dir_cleaned  = base_dir / "cleaned"
    dir_original.mkdir(parents=True, exist_ok=True)
    dir_cleaned.mkdir(parents=True, exist_ok=True)

    logger(f"[F1] Base dir: {base_dir}")
    logger(f"[F1] PDF: {pdf_path.name}")
    logger(f"[F1] Parametri: dpi={dpi}, anti_grid={anti_grid}, s_thr={s_thr}, v_thr={v_thr}, thicken={thicken}")

    try:
        doc = fitz.open(pdf_path.as_posix())
    except Exception as e:
        logger(f"[F1][ERR] Impossibile aprire il PDF: {e}")
        return

    logger(f"[F1] Pagine nel PDF: {doc.page_count}")

    pages_ok = 0
    for i, page in enumerate(doc, start=1):
        try:
            pix = page.get_pixmap(dpi=dpi)
            # buffer → ndarray
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # salva original
            out_orig = dir_original / f"page_{i:03d}.png"
            cv2.imwrite(out_orig.as_posix(), img)

            # genera cleaned con lo stesso identico algoritmo del .pyw
            if anti_grid:
                clean = _hsv_antigrid_keep_ink_bgr(img, s_thr=s_thr, v_thr=v_thr, thicken=thicken)
            else:
                clean = img

            out_clean = dir_cleaned / f"page_{i:03d}.png"
            cv2.imwrite(out_clean.as_posix(), clean)

            pages_ok += 1
            logger(f"[F1] Salvate original/cleaned pagina {i:03d}")
        except Exception as e:
            logger(f"[F1][WARN] Problema nella pagina {i}: {e}")

    doc.close()
    logger(f"[F1] Completato. Pagine esportate: {pages_ok}/{i if 'i' in locals() else 0}")


def _hsv_antigrid_keep_ink_bgr(img_bgr, s_thr=0.26, v_thr=0.65, thicken=2):
    """
    Porting 1:1 della tua funzione nel .pyw:
    - lavora in spazio 'HSV' calcolato a mano (senza cvtColor)
    - 'keep' = (S > s_thr) OR (V < v_thr)  → preserva inchiostro/zone scure e colori saturi
    - il resto va bianco
    - opzionale: dilata 'keep' per non perdere bordi (thicken)
    """
    import cv2, numpy as np

    img = img_bgr.astype(np.float32) / 255.0
    cmax = img.max(axis=2)                 # V
    cmin = img.min(axis=2)
    delta = cmax - cmin + 1e-6
    s = delta / (cmax + 1e-6)              # S
    v = cmax                               # V

    keep = (s > float(s_thr)) | (v < float(v_thr))

    if thicken and thicken > 1:
        keep = cv2.dilate(keep.astype(np.uint8),
                          np.ones((thicken, thicken), np.uint8),
                          iterations=1).astype(bool)

    clean = np.ones_like(img)              # bianco
    clean[keep] = img[keep]                # conserva solo i pixel 'keep'
    out = (np.clip(clean, 0, 1) * 255.0).astype(np.uint8)
    return out
