from pathlib import Path

def run_fase2(base_dir: Path, params: dict, logger=print):
    try:
        import cv2, numpy as np
    except Exception as e:
        logger(f"[F2][ERR] OpenCV non disponibile: {e}")
        logger("[F2] Installa con: pip install opencv-python")
        return

    mode        = str(params.get("mode", "otsu")).lower()
    fixed_thr   = float(params.get("fixed_thr", 0.65))     # [0..1]
    sau_w       = int(params.get("sau_w", 25))
    sau_k       = float(params.get("sau_k", 0.2))
    specks      = int(params.get("specks", 0))             # px per apertura su inchiostro
    dilate_px   = int(params.get("dilate", 0))             # px per dilatare inchiostro
    preview     = bool(params.get("preview", False))

    dir_cleaned = base_dir / "cleaned"
    dir_bw      = base_dir / "bw"
    dir_prev    = base_dir / "bw_preview"
    dir_bw.mkdir(parents=True, exist_ok=True)
    if preview: dir_prev.mkdir(parents=True, exist_ok=True)

    files = sorted(dir_cleaned.glob("*.png"))
    if not files:
        logger("[F2] Nessuna immagine in cleaned/. Esegui prima la Fase 1.")
        return

    logger(f"[F2] Avvio binarizzazione: mode={mode}, fixed={fixed_thr}, sau_w={sau_w}, sau_k={sau_k}, specks={specks}, dilate={dilate_px}, preview={preview}")

    # Sauvola on-demand
    sau_available = True
    if mode == "sauvola":
        try:
            from skimage.filters import threshold_sauvola
        except Exception:
            sau_available = False
            logger("[F2][WARN] skimage non disponibile: passo a Otsu.")
            mode = "otsu"

    cnt = 0
    for p in files:
        try:
            img = cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger(f"[F2][WARN] Immagine non leggibile: {p.name}")
                continue

            gray = img.astype("float32") / 255.0

            # 1) Trova la MASCHERA DELL’INCHIOSTRO (True dove ci sono tratti scuri)
            if mode == "fixed":
                ink_mask = (gray < fixed_thr)
            elif mode == "sauvola" and sau_available:
                th_map = threshold_sauvola(gray, window_size=max(3, sau_w), k=sau_k, r=None)
                ink_mask = (gray < th_map)
            else:  # otsu (default/fallback)
                thr_val, _ = cv2.threshold((gray*255).astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ink_mask = ((gray*255).astype("uint8") < thr_val)

            # 2) Operazioni morfologiche SULL’INCHIOSTRO (non sul bianco/nero finale)
            ink_u8 = (ink_mask.astype("uint8") * 255)
            if specks > 0:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (specks, specks))
                ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, k, iterations=1)  # rimuove puntini
            if dilate_px > 0:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
                ink_u8 = cv2.dilate(ink_u8, k, iterations=1)  # ispessisce tratti sottili
            ink_mask = (ink_u8 > 127)

            # 3) Costruisci l’immagine BW con SFONDO BIANCO e INCHIOSTRO NERO
            bw = np.where(ink_mask, 0, 255).astype("uint8")

            # 4) Salvare
            outp = (dir_bw / p.name).as_posix()
            cv2.imwrite(outp, bw)
            cnt += 1
            logger(f"[F2] bw: {p.name}")

            if preview:
                prevp = (dir_prev / p.name).as_posix()
                cv2.imwrite(prevp, bw)

        except Exception as e:
            logger(f"[F2][WARN] {p.name}: {e}")

    logger(f"[F2] Completato. File elaborati: {cnt}/{len(files)}")
