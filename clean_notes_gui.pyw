#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Clean Notes GUI — PDF ➜ Images ➜ B/W ➜ Skeleton ➜ Vector Strokes (JSON)

Fase 1: PDF -> immagini (original) con anti‑griglia opzionale -> cleaned
Fase 2: cleaned -> bianco/nero (bw) con Otsu/Sauvola
Fase 3: bw -> skeleton (1px) via skimage (skeletonize/medial_axis) -> bw_skel
Fase 4: skeleton -> vettori (polilinee) con grafo 8‑connesso, RDP semplificazione, sampling -> strokes/*.json

Dipendenze:
  pip install pymupdf opencv-python numpy scikit-image pillow

Note:
- Il codice è standalone e non dipende dal tuo file precedente. Salva output in sottocartelle della cartella scelta.
- JSON di output: {"width":W,"height":H,"strokes":[{"type":"polyline","closed":bool,"points":[[x,y],...]},...]}
"""

import os, json, math, threading
from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image

# -----------------------
# Utilities
# -----------------------

def ensure_dirs(base: Path):
    (base / "original").mkdir(parents=True, exist_ok=True)
    (base / "cleaned").mkdir(parents=True, exist_ok=True)
    (base / "bw").mkdir(parents=True, exist_ok=True)
    (base / "bw_skel").mkdir(parents=True, exist_ok=True)
    (base / "strokes").mkdir(parents=True, exist_ok=True)

def log_append(text_widget: tk.Text, msg: str):
    text_widget.insert("end", msg + "\n")
    text_widget.see("end")
    text_widget.update_idletasks()

# -----------------------
# Fase 1: PDF ➜ Immagini + Anti-griglia
# -----------------------

def hsv_antigrid_keep_ink_bgr(img_bgr: np.ndarray, s_thr=0.26, v_thr=0.65, thicken=2) -> np.ndarray:
    """Preserva inchiostro (S alta o V basso) e schiarisce il resto."""
    img = img_bgr.astype(np.float32) / 255.0
    cmax = img.max(axis=2)
    cmin = img.min(axis=2)
    delta = cmax - cmin + 1e-6
    s = delta / (cmax + 1e-6)
    v = cmax
    keep = (s > float(s_thr)) | (v < float(v_thr))
    if thicken and thicken > 1:
        keep = cv2.dilate(keep.astype(np.uint8), np.ones((thicken, thicken), np.uint8), iterations=1).astype(bool)
    clean = np.ones_like(img)
    clean[keep] = img[keep]
    out = (np.clip(clean, 0, 1) * 255.0).astype(np.uint8)
    return out

def pdf_to_images(pdf_path: Path, dpi: int, base: Path, anti_grid: bool, s_thr: float, v_thr: float, thicken: int, logw: tk.Text):
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        log_append(logw, f"[ERR] PyMuPDF mancante: {e}")
        return
    ensure_dirs(base)
    doc = fitz.open(pdf_path.as_posix())
    log_append(logw, f"[F1] Pagine PDF: {doc.page_count}")
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        orig_p = base / "original" / f"page_{i:03d}.png"
        cv2.imwrite(orig_p.as_posix(), img)
        if anti_grid:
            clean = hsv_antigrid_keep_ink_bgr(img, s_thr=s_thr, v_thr=v_thr, thicken=thicken)
        else:
            clean = img
        clean_p = base / "cleaned" / f"page_{i:03d}.png"
        cv2.imwrite(clean_p.as_posix(), clean)
        log_append(logw, f"[F1] Salvate original/cleaned pagina {i}")
    log_append(logw, "[F1] Completato.")

# -----------------------
# Fase 2: Binarizzazione
# -----------------------

def auto_white_background(bw_0_255: np.ndarray) -> np.ndarray:
    h, w = bw_0_255.shape[:2]
    border = np.concatenate([
        bw_0_255[0, :], bw_0_255[-1, :], bw_0_255[:, 0], bw_0_255[:, -1]
    ])
    if border.mean() < 127:
        bw_0_255 = cv2.bitwise_not(bw_0_255)
    return bw_0_255

def sauvola_threshold(gray01: np.ndarray, window: int = 25, k: float = 0.2, R: float = 0.5) -> np.ndarray:
    """Mappa di soglia Sauvola su float [0..1]."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(gray01, size=window, mode='reflect')
    mean_sq = uniform_filter(gray01 * gray01, size=window, mode='reflect')
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    th = mean * (1 + k * (std / (R + 1e-6) - 1))
    return th

def cleaned_to_bw(base: Path, mode: str, fixed: float, sau_w: int, sau_k: float, specks: int, dilate_px: int, logw: tk.Text):
    ensure_dirs(base)
    inp = base / "cleaned"
    out = base / "bw"
    files = sorted([p for p in inp.glob("*.png")])
    if not files:
        log_append(logw, "[F2] Nessuna immagine in cleaned/")
        return
    for p in files:
        img = cv2.imread(p.as_posix(), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if mode == "fixed":
            bw = (gray < float(fixed)).astype(np.uint8) * 255
        elif mode == "otsu":
            g8 = (gray * 255).astype(np.uint8)
            _, bw = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw = (255 - bw)  # inchiostro nero
        else:
            th_map = sauvola_threshold(gray, window=int(sau_w), k=float(sau_k), R=0.5)
            bw = ((gray < th_map).astype(np.uint8)) * 255

        if specks and specks > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (specks, specks))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        if dilate_px and dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
            bw = cv2.dilate(bw, k, iterations=1)

        bw = auto_white_background(bw)
        cv2.imwrite((out / p.name).as_posix(), bw)
        log_append(logw, f"[F2] bw: {p.name}")
    log_append(logw, "[F2] Completato.")

# -----------------------
# Fase 3: Skeleton
# -----------------------

try:
    from skimage.morphology import skeletonize as sk_skeletonize, medial_axis, remove_small_objects, binary_closing, square
    _HAS_SK = True
except Exception:
    _HAS_SK = False

def bw_to_skeleton(base: Path, method: str, min_obj: int, do_close: bool, logw: tk.Text):
    if not _HAS_SK:
        log_append(logw, "[F3] scikit-image non disponibile.")
        return
    ensure_dirs(base)
    inp = base / "bw"
    out = base / "bw_skel"
    files = sorted([p for p in inp.glob("*.png")])
    if not files:
        log_append(logw, "[F3] Nessun file in bw/")
        return
    for p in files:
        bw = cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE)
        mask = (bw < 128)
        if do_close:
            mask = binary_closing(mask, square(3))
        if min_obj and min_obj > 1:
            mask = remove_small_objects(mask, min_size=int(min_obj))
        if method == "skeletonize":
            sk = sk_skeletonize(mask)
        else:
            sk, _ = medial_axis(mask, return_distance=True)
        sk_img = (sk.astype(np.uint8) * 255)
        cv2.imwrite((out / p.name).as_posix(), 255 - sk_img)  # nero=tratto
        log_append(logw, f"[F3] skeleton: {p.name}")
    log_append(logw, "[F3] Completato.")

# -----------------------
# Fase 4: Vectorization (Skeleton -> Strokes JSON)
# -----------------------

@dataclass
class VecParams:
    dp_tol: float = 1.0      # tolleranza Douglas–Peucker in px
    min_points: int = 8      # scarta polilinee troppo corte
    close_gap: int = 2       # ricuce gap (px) tra endpoint
    sample_step: float = 1.5 # distanza target tra punti consecutivi

def neighbors8(y, x, H, W):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: 
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx

def build_graph_from_skeleton(mask: np.ndarray, close_gap: int):
    """mask True=ink skeleton. Restituisce: coords list, adjacency dict {idx: set(idx)}"""
    H, W = mask.shape
    coords = []               # idx -> (y,x)
    index = -np.ones_like(mask, dtype=np.int32)  # -1=non nodo
    ys, xs = np.where(mask)
    idx = 0
    for y, x in zip(ys, xs):
        index[y, x] = idx
        coords.append((int(y), int(x)))
        idx += 1
    coords = np.array(coords, dtype=np.int32)
    adj = {i: set() for i in range(idx)}
    for y, x in zip(ys, xs):
        i = int(index[y, x])
        for ny, nx in neighbors8(y, x, H, W):
            if index[ny, nx] >= 0:
                j = int(index[ny, nx])
                if i != j:
                    adj[i].add(j)
    # gap closing: collega endpoint vicini entro r
    if close_gap and close_gap > 0:
        endpoints = [i for i in range(idx) if len(adj[i]) == 1]
        # naive: k-d tree semplice con numpy
        ep_coords = coords[endpoints] if endpoints else np.zeros((0,2), np.int32)
        for a, ia in enumerate(endpoints):
            ya, xa = ep_coords[a]
            # cerca in un quadrato r
            y0, y1 = max(0, ya-close_gap), min(H-1, ya+close_gap)
            x0, x1 = max(0, xa-close_gap), min(W-1, xa+close_gap)
            sub = index[y0:y1+1, x0:x1+1]
            cand = np.unique(sub[sub>=0])
            for j in cand:
                if j == ia: 
                    continue
                # collega solo endpoint o pixel molto vicini
                yb, xb = coords[j]
                if (abs(ya - yb) + abs(xa - xb)) <= close_gap:
                    adj[ia].add(int(j))
                    adj[int(j)].add(ia)
    return coords, adj, index

def douglas_peucker(points, eps):
    """RDP su array Nx2 (y,x) — ritorna punti semplificati nell'ordine originale."""
    if len(points) < 3:
        return points
    # distanza punto-linea
    (y1, x1) = points[0]
    (y2, x2) = points[-1]
    vy = y2 - y1
    vx = x2 - x1
    vnorm = math.hypot(vy, vx) + 1e-9
    dmax = -1.0
    idx = -1
    for i in range(1, len(points)-1):
        py, px = points[i]
        # area parallelogramma / base
        dist = abs(vy*(px - x1) - vx*(py - y1)) / vnorm
        if dist > dmax:
            dmax = dist
            idx = i
    if dmax > eps:
        left = douglas_peucker(points[:idx+1], eps)
        right = douglas_peucker(points[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.array([points[0], points[-1]])

def resample_polyline(points, step):
    """Campiona lungo la lunghezza a passo ~step (in px)."""
    if len(points) <= 2 or step <= 0:
        return points
    pts = points.astype(np.float32)
    dists = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(dists)])
    L = cum[-1]
    if L <= step:
        return points
    targets = np.arange(0.0, L, step)
    out = []
    k = 0
    for t in targets:
        while not (cum[k] <= t <= cum[k+1]):
            k += 1
            if k >= len(dists):
                break
        if k >= len(dists):
            break
        a, b = pts[k], pts[k+1]
        seg = cum[k+1] - cum[k]
        alpha = 0.0 if seg == 0 else (t - cum[k]) / seg
        out.append(a + alpha*(b - a))
    out.append(pts[-1])
    return np.round(np.vstack(out)).astype(np.int32)

def trace_polylines_from_graph(coords: np.ndarray, adj: dict, params: VecParams):
    N = coords.shape[0]
    visited_edge = set()  # edges as tuple(min,max)
    deg = np.array([len(adj[i]) for i in range(N)], dtype=np.int32)
    endpoints = [i for i in range(N) if deg[i] == 1]
    junctions = set(i for i in range(N) if deg[i] >= 3)
    polylines = []

    def take_edge(a, b):
        e = (a, b) if a < b else (b, a)
        if e in visited_edge:
            return False
        visited_edge.add(e)
        return True

    def walk_from(start):
        # trova un vicino per partire
        if not adj[start]:
            return
        path = [start]
        # scegli il primo vicino non visitato
        nxts = sorted(list(adj[start]), key=lambda j: abs(j-start))
        prev = start
        cur = nxts[0]
        if not take_edge(prev, cur):
            return
        path.append(cur)
        while True:
            # cerca prossimo vicino diverso da prev
            neigh = [k for k in adj[cur] if k != prev]
            # prediligi angolo più piccolo
            if len(neigh) == 0:
                break
            best = None
            if len(neigh) == 1:
                cand = neigh[0]
            else:
                # minimizza deviazione angolare
                v1 = coords[cur] - coords[prev]
                ang_best = 1e9
                cand = neigh[0]
                for nb in neigh:
                    v2 = coords[nb] - coords[cur]
                    num = float(np.dot(v1, v2))
                    den = (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
                    cosang = -num/den  # vogliamo deviazione minima => cos vicino a -1 (continuare dritto)
                    if cosang < ang_best:
                        ang_best = cosang
                        cand = nb
            if not take_edge(cur, cand):
                break
            path.append(cand)
            prev, cur = cur, cand
            if cur in junctions:
                break
        polylines.append(path)

    # 1) percorsi da endpoint
    for s in endpoints:
        # controlla se ha ancora edge liberi
        free = [j for j in adj[s] if ((min(s,j), max(s,j)) not in visited_edge)]
        if free:
            walk_from(s)

    # 2) loop residui (nessun endpoint)
    for i in range(N):
        for j in adj[i]:
            e = (min(i,j), max(i,j))
            if e not in visited_edge:
                # traccia piccolo ciclo
                path = [i, j]
                visited_edge.add(e)
                prev, cur = i, j
                while True:
                    neigh = [k for k in adj[cur] if k != prev]
                    if not neigh:
                        break
                    cand = neigh[0]
                    e2 = (min(cur, cand), max(cur, cand))
                    if e2 in visited_edge:
                        break
                    visited_edge.add(e2)
                    path.append(cand)
                    prev, cur = cur, cand
                    if cand == i:
                        break
                polylines.append(path)

    # converti indici in coordinate e applica pulizia/semplificazione
    out = []
    for path in polylines:
        if len(path) < params.min_points:
            continue
        pts = coords[np.array(path, dtype=np.int32)][:, ::-1]  # (x,y) per comodità output
        # RDP in (x,y) ma la nostra distanza usa (y,x); convertiamo
        pts_yx = pts[:, ::-1]  # (y,x) per coerenza con resample
        simp = douglas_peucker(pts_yx, params.dp_tol)
        rs = resample_polyline(simp, params.sample_step)
        if len(rs) < params.min_points:
            continue
        # chiusura se primo/ultimo vicini
        closed = np.linalg.norm(rs[0] - rs[-1]) <= max(2, int(params.dp_tol*2))
        # output come (x,y)
        rs_xy = rs[:, ::-1].tolist()
        out.append({"type": "polyline", "closed": bool(closed), "points": rs_xy})
    return out

def skeletons_to_strokes(base: Path, params: VecParams, logw: tk.Text):
    ensure_dirs(base)
    inp = base / "bw_skel"
    out_dir = base / "strokes"
    files = sorted([p for p in inp.glob("*.png")])
    if not files:
        log_append(logw, "[F4] Nessun file in bw_skel/")
        return
    for p in files:
        sk = cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE)
        # inchiostro nero -> mask True
        mask = (sk < 128)
        H, W = mask.shape
        coords, adj, _ = build_graph_from_skeleton(mask, close_gap=params.close_gap)
        strokes = trace_polylines_from_graph(coords, adj, params)
        payload = {
            "width": int(W),
            "height": int(H),
            "strokes": strokes
        }
        outp = out_dir / (p.stem + ".json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
        kb = (outp.stat().st_size + 1023)//1024
        log_append(logw, f"[F4] strokes: {outp.name} ({kb} KB, {len(strokes)} stroke)")
    log_append(logw, "[F4] Completato.")

# -----------------------
# GUI
# -----------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clean Notes GUI — Fasi 1-4")
        self.geometry("980x720")
        self.minsize(900, 650)

        self.base_dir = tk.StringVar(value=str(Path.home() / "CleanNotesOut"))
        self.pdf_path = tk.StringVar(value="")
        self.dpi = tk.IntVar(value=600)
        self.anti_grid = tk.BooleanVar(value=True)
        self.s_thr = tk.DoubleVar(value=0.26)
        self.v_thr = tk.DoubleVar(value=0.65)
        self.thicken = tk.IntVar(value=2)

        self.mode = tk.StringVar(value="otsu")
        self.fixed = tk.DoubleVar(value=0.65)
        self.sau_w = tk.IntVar(value=25)
        self.sau_k = tk.DoubleVar(value=0.2)
        self.specks = tk.IntVar(value=0)
        self.dilate = tk.IntVar(value=0)

        self.sk_method = tk.StringVar(value="skeletonize")
        self.min_obj = tk.IntVar(value=16)
        self.do_close = tk.BooleanVar(value=False)

        self.dp_tol = tk.DoubleVar(value=1.0)
        self.min_points = tk.IntVar(value=8)
        self.close_gap = tk.IntVar(value=2)
        self.sample_step = tk.DoubleVar(value=1.5)

        self._build_ui()

    def _pick_dir(self):
        d = filedialog.askdirectory(initialdir=self.base_dir.get() or str(Path.home()))
        if d:
            self.base_dir.set(d)

    def _pick_pdf(self):
        f = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if f:
            self.pdf_path.set(f)

    def _run_thread(self, fn, *args):
        th = threading.Thread(target=fn, args=args, daemon=True)
        th.start()

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        # ------------- Fase 1 -------------
        f1 = ttk.Frame(nb)
        nb.add(f1, text="Fase 1: PDF ➜ cleaned")
        frm = ttk.LabelFrame(f1, text="Parametri")
        frm.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm, text="Cartella di output").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=self.base_dir, width=60).grid(row=0, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(frm, text="Sfoglia…", command=self._pick_dir).grid(row=0, column=2, padx=6, pady=4)

        ttk.Label(frm, text="PDF").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=self.pdf_path, width=60).grid(row=1, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(frm, text="Apri PDF…", command=self._pick_pdf).grid(row=1, column=2, padx=6, pady=4)

        ttk.Label(frm, text="DPI").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm, from_=150, to=1200, increment=50, textvariable=self.dpi, width=10).grid(row=2, column=1, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(frm, text="Anti‑griglia HSV", variable=self.anti_grid).grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm, text="S soglia").grid(row=4, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=self.s_thr, width=8).grid(row=4, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm, text="V soglia").grid(row=5, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=self.v_thr, width=8).grid(row=5, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm, text="Thicken").grid(row=6, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm, from_=0, to=7, textvariable=self.thicken, width=8).grid(row=6, column=1, sticky="w", padx=6, pady=4)

        self.log1 = tk.Text(f1, height=18)
        self.log1.pack(fill="both", expand=True, padx=10, pady=8)

        ttk.Button(f1, text="Esegui Fase 1", command=lambda: self._run_thread(
            pdf_to_images, Path(self.pdf_path.get()), int(self.dpi.get()),
            Path(self.base_dir.get()), bool(self.anti_grid.get()),
            float(self.s_thr.get()), float(self.v_thr.get()), int(self.thicken.get()),
            self.log1)).pack(pady=6)

        # ------------- Fase 2 -------------
        f2 = ttk.Frame(nb)
        nb.add(f2, text="Fase 2: cleaned ➜ bw")
        frm2 = ttk.LabelFrame(f2, text="Parametri")
        frm2.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm2, text="Metodo").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Combobox(frm2, values=["fixed","otsu","sauvola"], textvariable=self.mode, state="readonly").grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm2, text="Soglia fissa").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm2, textvariable=self.fixed, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm2, text="Sauvola win").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm2, from_=9, to=101, increment=2, textvariable=self.sau_w, width=8).grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm2, text="Sauvola k").grid(row=3, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm2, textvariable=self.sau_k, width=8).grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm2, text="Remove specks").grid(row=4, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm2, from_=0, to=7, textvariable=self.specks, width=8).grid(row=4, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm2, text="Dilate px").grid(row=5, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm2, from_=0, to=5, textvariable=self.dilate, width=8).grid(row=5, column=1, sticky="w", padx=6, pady=4)

        self.log2 = tk.Text(f2, height=18)
        self.log2.pack(fill="both", expand=True, padx=10, pady=8)

        ttk.Button(f2, text="Esegui Fase 2", command=lambda: self._run_thread(
            cleaned_to_bw, Path(self.base_dir.get()), self.mode.get(), float(self.fixed.get()),
            int(self.sau_w.get()), float(self.sau_k.get()), int(self.specks.get()),
            int(self.dilate.get()), self.log2
        )).pack(pady=6)

        # ------------- Fase 3 -------------
        f3 = ttk.Frame(nb)
        nb.add(f3, text="Fase 3: bw ➜ skeleton")
        frm3 = ttk.LabelFrame(f3, text="Parametri")
        frm3.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm3, text="Metodo").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Combobox(frm3, values=["skeletonize","medial_axis"], textvariable=self.sk_method, state="readonly").grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm3, text="Min object").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm3, from_=0, to=256, textvariable=self.min_obj, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(frm3, text="Binary closing 3×3", variable=self.do_close).grid(row=2, column=1, sticky="w", padx=6, pady=4)

        self.log3 = tk.Text(f3, height=18)
        self.log3.pack(fill="both", expand=True, padx=10, pady=8)

        ttk.Button(f3, text="Esegui Fase 3", command=lambda: self._run_thread(
            bw_to_skeleton, Path(self.base_dir.get()), self.sk_method.get(),
            int(self.min_obj.get()), bool(self.do_close.get()), self.log3
        )).pack(pady=6)

        # ------------- Fase 4 -------------
        f4 = ttk.Frame(nb)
        nb.add(f4, text="Fase 4: skeleton ➜ strokes (JSON)")
        frm4 = ttk.LabelFrame(f4, text="Parametri")
        frm4.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm4, text="RDP tol (px)").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm4, textvariable=self.dp_tol, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm4, text="Min points").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm4, from_=2, to=128, textvariable=self.min_points, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm4, text="Close gap (px)").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm4, from_=0, to=6, textvariable=self.close_gap, width=8).grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm4, text="Sample step (px)").grid(row=3, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm4, textvariable=self.sample_step, width=8).grid(row=3, column=1, sticky="w", padx=6, pady=4)

        self.log4 = tk.Text(f4, height=18)
        self.log4.pack(fill="both", expand=True, padx=10, pady=8)

        ttk.Button(f4, text="Esegui Fase 4", command=lambda: self._run_thread(
            skeletons_to_strokes, Path(self.base_dir.get()),
            VecParams(dp_tol=float(self.dp_tol.get()),
                      min_points=int(self.min_points.get()),
                      close_gap=int(self.close_gap.get()),
                      sample_step=float(self.sample_step.get())),
            self.log4
        )).pack(pady=6)

if __name__ == "__main__":
    App().mainloop()
