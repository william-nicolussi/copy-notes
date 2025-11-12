# core/skeletonize_.py
from pathlib import Path

def run_fase3(base_dir: Path, params: dict, logger=print):
    """
    Legge PNG da base/bw, skeletonizza l'inchiostro (nero) e salva in base/bw_skel.
    Parametri:
      - method: "skeletonize" | "medial_axis"
      - min_obj: rimuove componenti più piccole di min_obj pixel (0 = disattivo)
      - do_close: True/False, closing 3x3 prima della schelettrizzazione
      - preview: True/False, salva anche anteprima colorata in bw_skel_preview
    """
    try:
        import cv2, numpy as np
    except Exception as e:
        logger(f"[F3][ERR] OpenCV non disponibile: {e}")
        return

    # scikit-image per skeleton
    try:
        from skimage.morphology import skeletonize, medial_axis, remove_small_objects, binary_closing, square
    except Exception as e:
        logger(f"[F3][ERR] scikit-image non disponibile ({e}). Installa con: pip install scikit-image")
        return

    method   = str(params.get("method", "skeletonize")).lower()
    min_obj  = int(params.get("min_obj", 16))
    do_close = bool(params.get("do_close", False))
    preview  = bool(params.get("preview", False))

    dir_bw   = base_dir / "bw"
    dir_out  = base_dir / "bw_skel"
    dir_prev = base_dir / "bw_skel_preview"
    dir_out.mkdir(parents=True, exist_ok=True)
    if preview:
        dir_prev.mkdir(parents=True, exist_ok=True)

    files = sorted(dir_bw.glob("*.png"))
    if not files:
        logger("[F3] Nessun file in bw/. Esegui prima la Fase 2.")
        return

    logger(f"[F3] Avvio skeleton: method={method}, min_obj={min_obj}, closing={do_close}, preview={preview}")
    cnt = 0

    for p in files:
        try:
            bw = cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE)
            if bw is None:
                logger(f"[F3][WARN] Immagine non leggibile: {p.name}")
                continue

            # ink = nero (0) → True
            ink = (bw < 128)

            # pre-proc opzionale
            if do_close:
                ink = binary_closing(ink, square(3))
            if min_obj > 0:
                ink = remove_small_objects(ink, min_size=min_obj)

            # skeleton
            if method == "medial_axis":
                skel, _ = medial_axis(ink, return_distance=True)
            else:
                skel = skeletonize(ink)

            # output: nero su bianco
            skel_img = (skel == False).astype("uint8") * 255  # True→0 (nero), False→255 (bianco)
            outp = (dir_out / p.name).as_posix()
            cv2.imwrite(outp, skel_img)
            cnt += 1
            logger(f"[F3] skeleton: {p.name}")

            # preview opzionale (linee rosse su sfondo bianco)
            if preview:
                preview_img = cv2.cvtColor(skel_img, cv2.COLOR_GRAY2BGR)
                preview_img[skel] = (0, 0, 255)
                prev_path = dir_prev / p.name
                cv2.imwrite(prev_path.as_posix(), preview_img)

        except Exception as e:
            logger(f"[F3][WARN] {p.name}: {e}")

    logger(f"[F3] Completato. File elaborati: {cnt}/{len(files)}")
