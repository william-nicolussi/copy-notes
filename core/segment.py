# core/segment.py
from pathlib import Path
import json

def run_fase4(base_dir: Path, params: dict, logger=print):
    """
    Legge skeleton PNG (nero su bianco) da base/bw_skel,
    estrae strokes (cammini massimi su 8-connettività) + dots isolati,
    salva JSON in base/strokes e anteprima opzionale in base/strokes_preview.
    """
    try:
        import cv2, numpy as np
    except Exception as e:
        logger(f"[F4][ERR] OpenCV non disponibile: {e}")
        return

    preview = bool(params.get("preview", True))

    dir_in   = base_dir / "bw_skel"
    dir_out  = base_dir / "strokes"
    dir_prev = base_dir / "strokes_preview"
    dir_out.mkdir(parents=True, exist_ok=True)
    if preview:
        dir_prev.mkdir(parents=True, exist_ok=True)

    files = sorted(dir_in.glob("*.png"))
    if not files:
        logger("[F4] Nessun file in bw_skel/. Esegui prima la Fase 3.")
        return

    logger(f"[F4] Avvio segmentazione: preview={preview}")
    done = 0

    for p in files:
        try:
            img = cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger(f"[F4][WARN] Immagine non leggibile: {p.name}")
                continue

            # mask True sui pixel neri (inchiostro)
            ink = (img < 128)
            h, w = ink.shape

            strokes, dots = _extract_strokes_from_mask(ink)

            # salva JSON
            payload = {"width": int(w), "height": int(h), "strokes": strokes, "dots": dots}
            out_json = (dir_out / (p.stem + ".json"))
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

            kb = (out_json.stat().st_size + 1023) // 1024
            logger(f"[F4] strokes: {out_json.name} ({kb} KB, {len(strokes)} stroke, {len(dots)} dots)")
            done += 1

            # preview opzionale
            if preview:
                prev = _draw_preview(h, w, strokes, dots)
                cv2.imwrite((dir_prev / (p.stem + "_preview.png")).as_posix(), prev)

        except Exception as e:
            logger(f"[F4][WARN] {p.name}: {e}")

    logger(f"[F4] Completato. File elaborati: {done}/{len(files)}")


def _extract_strokes_from_mask(mask):
    """
    Input: mask (bool) True sugli "ink" pixel (scheletro 1px).
    Output:
      strokes: list of list of [x,y]
      dots:    list of [x,y]
    Strategia:
      - mappa id pixel -> grado (8-neighborhood)
      - nodi = pixel con grado != 2
      - percorri cammini massimi tra nodi
      - gestisci componenti "lineari" chiuse o open che non toccano nodi (tutti gradi==2)
      - dots = componenti isolate (grado==0)
    """
    import numpy as np

    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [], []

    # mapping (y,x) -> id e back
    ids = np.arange(len(xs), dtype=np.int32)
    key = np.stack([ys, xs], axis=1)
    # hash per lookup rapido
    lut = {}
    for i, (yy, xx) in enumerate(key):
        lut[(int(yy), int(xx))] = i

    # 8-neighborhood offsets
    OFF = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # calcola adiacenze e grado
    degree = np.zeros(len(xs), dtype=np.uint8)
    nbrs = [[] for _ in range(len(xs))]
    for i, (yy, xx) in enumerate(key):
        d = 0
        nn = []
        for dy, dx in OFF:
            ny, nx = yy+dy, xx+dx
            j = lut.get((int(ny), int(nx)))
            if j is not None:
                nn.append(j)
                d += 1
        degree[i] = d
        nbrs[i] = nn

    visited = np.zeros(len(xs), dtype=bool)

    # helper per trasformare lista di id -> lista [x,y] (in ordine in cui si visita)
    def ids_to_xy(seq_ids):
        # convertiamo in [x,y] come richiesto
        pts = []
        for i in seq_ids:
            yy, xx = key[i]
            pts.append([int(xx), int(yy)])
        return pts

    strokes = []
    dots = []

    # 1) Dots: componenti con grado 0 (pixel isolati)
    for i in range(len(xs)):
        if degree[i] == 0 and not visited[i]:
            visited[i] = True
            dots.append([int(key[i][1]), int(key[i][0])])  # [x,y]

    # 2) Cammini che partono da "nodi" (grado != 2)
    def walk_from(u, prev=None):
        """
        cammino massimo partendo da u (che è nodo o endpoint),
        avanza finché trova catena di gradi==2, si ferma al prossimo nodo/endpoint
        ritorna lista di ids visitati nell'ordine
        """
        path = [u]
        visited[u] = True
        cur = u
        prev_id = prev

        while True:
            # scegli prossima tra i vicini non-visitati (diversa da prev)
            nexts = [v for v in nbrs[cur] if (not visited[v]) and (v != prev_id)]
            if not nexts:
                break
            if len(nexts) > 1:
                # biforcazione interna inattesa in catena; prendiamo un vicino e le altre usciranno da altri walk
                nxt = nexts[0]
            else:
                nxt = nexts[0]

            # regola: fermati se cur è nodo (grado !=2) e hai già almeno 1 passo (per non saltare la prima mossa)
            if cur != u and degree[cur] != 2:
                break

            path.append(nxt)
            visited[nxt] = True
            prev_id, cur = cur, nxt

            # se arrivo a un vero nodo, chiudo
            if degree[cur] != 2:
                break

        return path

    # parti da nodi ed endpoint
    node_idxs = [i for i in range(len(xs)) if degree[i] != 2 and not visited[i] and degree[i] > 0]
    for u in node_idxs:
        if visited[u]:  # può essere marcato da un cammino vicino
            continue
        # da un nodo, esplora tutti i vicini non visitati come rami
        for v in [vv for vv in nbrs[u] if not visited[vv]]:
            # avvia cammino nel senso (u -> v)
            visited[u] = True
            path_uv = [u]
            cur, prev_id = v, u
            path_uv.append(cur)
            visited[cur] = True

            while True:
                # se il corrente è nodo (e non è l'origine), chiudi
                if degree[cur] != 2 and cur != u:
                    break
                # trova prossimo
                nxts = [w for w in nbrs[cur] if (w != prev_id)]
                if not nxts:
                    break
                # in catena ci sarà al massimo 1 prossimo non prev
                nxt = nxts[0] if len(nxts) == 1 else nxts[0]
                if visited[nxt] and degree[nxt] == 2:
                    break
                path_uv.append(nxt)
                visited[nxt] = True
                prev_id, cur = cur, nxt

            if len(path_uv) >= 2:
                strokes.append(ids_to_xy(path_uv))

    # 3) Componenti “cicliche” di soli gradi==2 non ancora visitate
    for i in range(len(xs)):
        if visited[i] or degree[i] != 2:
            continue
        # cammina finché non torni indietro o chiudi il ciclo
        cyc = []
        cur = i
        prev_id = None
        while True:
            if visited[cur]:
                break
            visited[cur] = True
            cyc.append(cur)
            nxts = [v for v in nbrs[cur] if v != prev_id]
            if not nxts:
                break
            # se c'è un vicino non visitato, prosegui
            nxt = nxts[0]
            prev_id, cur = cur, nxt
            if cur == i:
                # chiuso ciclo
                break
        if len(cyc) >= 2:
            strokes.append(ids_to_xy(cyc))

    return strokes, dots


def _draw_preview(h, w, strokes, dots):
    """
    Crea un'immagine RGB su sfondo bianco con:
      - strokes disegnati a colori
      - dots in rosso
    """
    import cv2, numpy as np
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # palette semplice
    colors = (
        (0, 0, 255),      # red
        (255, 0, 0),      # blue
        (0, 128, 0),      # green
        (0, 128, 255),    # orange-ish
        (255, 0, 255),    # magenta
        (128, 0, 128),    # purple
        (128, 128, 0),    # olive
        (0, 0, 0),        # black
    )

    # disegna strokes
    for idx, pts in enumerate(strokes):
        col = colors[idx % len(colors)]
        if len(pts) == 1:
            x, y = pts[0]
            cv2.circle(canvas, (x, y), 2, col, -1, lineType=cv2.LINE_AA)
        else:
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i+1]
                cv2.line(canvas, (x1, y1), (x2, y2), col, 1, lineType=cv2.LINE_AA)

    # disegna dots
    for x, y in dots:
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return canvas
