#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Clean Notes GUI — v4 SEG (segmentazione pura)
PDF ➜ Images ➜ B/W ➜ Skeleton ➜ Strokes (segmentazione senza semplificazioni)
- Fase 4 ora decompone lo skeleton in cammini massimi tra nodi non-2 (endpoint/giunzioni) e gestisce i "dot" isolati.
- Nessuna semplificazione geometrica, nessun resampling, nessun close-gap, nessun filtro lunghezza.
- Preview opzionale: overlay dei tratti su skeleton e copia delle BW.

Dipendenze:
  pip install pymupdf opencv-python numpy scikit-image pillow
"""

import os, json, threading
from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import cv2

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
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(gray01, size=window, mode='reflect')
    mean_sq = uniform_filter(gray01 * gray01, size=window, mode='reflect')
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    th = mean * (1 + k * (std / (R + 1e-6) - 1))
    return th

def cleaned_to_bw(base: Path, mode: str, fixed: float, sau_w: int, sau_k: float, specks: int, dilate_px: int, preview_bw: bool, logw: tk.Text):
    ensure_dirs(base)
    inp = base / "cleaned"
    out = base / "bw"
    if preview_bw:
        (base / "bw_preview").mkdir(parents=True, exist_ok=True)
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
            bw = (255 - bw)
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
        outp = (out / p.name)
        cv2.imwrite(outp.as_posix(), bw)
        if preview_bw:
            prevp = (base / "bw_preview" / p.name)
            cv2.imwrite(prevp.as_posix(), bw)
        log_append(logw, f"[F2] bw: {p.name}" + (" [preview]" if preview_bw else ""))
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
        cv2.imwrite((out / p.name).as_posix(), 255 - sk_img)
        log_append(logw, f"[F3] skeleton: {p.name}")
    log_append(logw, "[F3] Completato.")

# -----------------------
# Fase 4: Segmentazione pura (Skeleton -> Strokes JSON)
# -----------------------

def neighbors8(y, x, H, W):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx

def build_graph(mask: np.ndarray):
    """mask True=ink. Restituisce coords (N,2: y,x), adj dict, idx_map"""
    H, W = mask.shape
    ys, xs = np.where(mask)
    N = len(ys)
    idx_map = -np.ones((H, W), dtype=np.int32)
    coords = np.empty((N, 2), dtype=np.int32)
    for i, (y, x) in enumerate(zip(ys, xs)):
        idx_map[y, x] = i
        coords[i] = (y, x)
    adj = {i: set() for i in range(N)}
    for i, (y, x) in enumerate(coords):
        for ny, nx in neighbors8(y, x, H, W):
            j = idx_map[ny, nx]
            if j >= 0 and j != i:
                adj[i].add(j)
    return coords, adj, idx_map

def decompose_paths(coords, adj):
    """Decompone il grafo in cammini massimi e loop; include DOT per isolati."""
    N = len(coords)
    deg = np.array([len(adj[i]) for i in range(N)], dtype=np.int32)
    visited = set()  # edges (min,max)
    paths = []       # lista di liste di indici
    dots = []        # indici isolati (deg==0)

    # Gestisci isolati
    for i in range(N):
        if deg[i] == 0:
            dots.append(i)

    def take(a, b):
        e = (a, b) if a < b else (b, a)
        if e in visited:
            return False
        visited.add(e)
        return True

    # Cammini da nodi con deg != 2 (endpoint e giunzioni)
    seeds = [i for i in range(N) if deg[i] != 2 and deg[i] > 0]
    for s in seeds:
        for nb in list(adj[s]):
            if not take(s, nb):
                continue
            path = [s, nb]
            prev, cur = s, nb
            while True:
                # prosegui finché sei in catena (deg==2)
                nexts = [k for k in adj[cur] if k != prev]
                if deg[cur] != 2 or len(nexts) == 0:
                    break
                nxt = nexts[0] if len(nexts) == 1 else nexts[0]  # non importa quale: copriamo tutto
                if not take(cur, nxt):
                    break
                path.append(nxt)
                prev, cur = cur, nxt
            paths.append(path)

    # Loop residui (componenti con tutti deg==2)
    for i in range(N):
        for j in adj[i]:
            e = (i, j) if i < j else (j, i)
            if e in visited:
                continue
            # avvia un giro lungo il loop
            take(i, j)
            path = [i, j]
            prev, cur = i, j
            while True:
                nexts = [k for k in adj[cur] if k != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                if not take(cur, nxt):
                    break
                path.append(nxt)
                prev, cur = cur, nxt
                if nxt == i:
                    break
            paths.append(path)

    return paths, dots

def draw_strokes_preview_on_skeleton(sk_path: Path, strokes: list, dots: list, out_path: Path,
                                     poly_color=(0,0,255), dot_color=(0,255,0), dot_radius=2, thickness=1):
    base = cv2.imread(sk_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if base is None:
        return
    color_img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # polilinee
    for st in strokes:
        pts = np.array(st["points"], dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(color_img, [pts], isClosed=bool(st.get("closed", False)),
                      color=poly_color, thickness=thickness, lineType=cv2.LINE_AA)
    # dots
    for d in dots:
        x, y = d["point"]
        cv2.circle(color_img, (int(x), int(y)), dot_radius, dot_color, -1, lineType=cv2.LINE_AA)
    cv2.imwrite(out_path.as_posix(), color_img)

def skeletons_to_strokes_segment(base: Path, preview_strokes: bool, logw: tk.Text):
    ensure_dirs(base)
    inp = base / "bw_skel"
    out_dir = base / "strokes"
    prev_dir = base / "strokes_preview"
    if preview_strokes:
        prev_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in inp.glob("*.png")])
    if not files:
        log_append(logw, "[F4] Nessun file in bw_skel/")
        return
    for p in files:
        sk = cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE)
        mask = (sk < 128)
        H, W = mask.shape
        coords, adj, _ = build_graph(mask)
        paths, dot_idx = decompose_paths(coords, adj)

        strokes = []
        for path in paths:
            # converti in (x,y)
            pts_xy = np.array([coords[i][::-1] for i in path], dtype=np.int32).tolist()
            closed = (len(path) >= 3 and path[0] == path[-1])
            strokes.append({"type": "polyline", "closed": bool(closed), "points": pts_xy})

        dots = []
        for i in dot_idx:
            x, y = int(coords[i][1]), int(coords[i][0])
            dots.append({"type": "dot", "point": [x, y]})

        payload = {
            "width": int(W),
            "height": int(H),
            "strokes": strokes,
            "dots": dots
        }
        outp = out_dir / (p.stem + ".json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)

        kb = (outp.stat().st_size + 1023)//1024
        log_append(logw, f"[F4] strokes: {outp.name} ({kb} KB, {len(strokes)} stroke, {len(dots)} dots)")

        if preview_strokes:
            prevp = prev_dir / (p.stem + "_preview.png")
            draw_strokes_preview_on_skeleton(p, strokes, dots, prevp,
                                             poly_color=(0,0,255), dot_color=(0,200,0),
                                             dot_radius=2, thickness=1)
    log_append(logw, "[F4] Completato.")

# -----------------------
# GUI
# -----------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clean Notes GUI — Fasi 1-4 (Segmentazione pura)")
        self.geometry("1000x760")
        self.minsize(920, 680)

        self.base_dir = tk.StringVar(value=str(Path(__file__).resolve().parent))
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
        self.preview_bw = tk.BooleanVar(value=False)

        self.sk_method = tk.StringVar(value="skeletonize")
        self.min_obj = tk.IntVar(value=16)
        self.do_close = tk.BooleanVar(value=False)

        self.preview_strokes = tk.BooleanVar(value=True)

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
        ttk.Checkbutton(frm2, text="See preview (bw_preview)", variable=self.preview_bw).grid(row=6, column=1, sticky="w", padx=6, pady=4)

        self.log2 = tk.Text(f2, height=18)
        self.log2.pack(fill="both", expand=True, padx=10, pady=8)

        ttk.Button(f2, text="Esegui Fase 2", command=lambda: self._run_thread(
            cleaned_to_bw, Path(self.base_dir.get()), self.mode.get(), float(self.fixed.get()),
            int(self.sau_w.get()), float(self.sau_k.get()), int(self.specks.get()),
            int(self.dilate.get()), bool(self.preview_bw.get()), self.log2
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
        nb.add(f4, text="Fase 4: skeleton ➜ strokes (SEG)")
        frm4 = ttk.LabelFrame(f4, text="Opzioni")
        frm4.pack(fill="x", padx=10, pady=8)

        self.preview_strokes = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm4, text="See preview (strokes_preview)", variable=self.preview_strokes).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        self.log4 = tk.Text(f4, height=18)
        self.log4.pack(fill="both", expand=True, padx=10, pady=8)

        ttk.Button(f4, text="Esegui Fase 4 (segmentazione pura)", command=lambda: self._run_thread(
            skeletons_to_strokes_segment, Path(self.base_dir.get()),
            bool(self.preview_strokes.get()), self.log4
        )).pack(pady=6)

if __name__ == "__main__":
    App().mainloop()
