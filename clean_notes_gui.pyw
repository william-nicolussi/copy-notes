#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Clean Notes GUI — PDF ➜ Images ➜ B/W ➜ Strokes
Versione completa con vettorizzazione edge‑based e log dettagliato.
- Fase 1: PDF -> immagini (original/cleaned) con anti‑griglia (HSV) opzionale
- Fase 2: cleaned -> bianco/nero (bw) con Otsu/Sauvola
- Fase 3: bw -> strokes JSON (edge‑based, 1‑px thinning, pruning rametti, RDP)

Dipendenze principali:
  pip install pymupdf opencv-python numpy

Nota: se preferisci mantenere i tuoi default di Fase 1/2, puoi copiare
solo la classe Stage3Frame in un tuo file esistente: è autosufficiente.
"""

import json
import math
import time
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2

# ============================
#  Helpers comuni
# ============================

def ensure_dirs(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    d_orig = base_dir / "original"
    d_clean = base_dir / "cleaned"
    d_bw = base_dir / "bw"
    d_strk = base_dir / "strokes"
    for d in (d_orig, d_clean, d_bw, d_strk):
        d.mkdir(exist_ok=True)
    return d_orig, d_clean, d_bw, d_strk

def list_images(folder: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])

def auto_white_background(bw_img):
    border = np.concatenate([bw_img[0,:], bw_img[-1,:], bw_img[:,0], bw_img[:,-1]])
    if border.mean() < 127:
        bw_img = cv2.bitwise_not(bw_img)
    return bw_img

# ============================
#  Fase 1: PDF -> original/cleaned
# ============================

def hsv_antigrid_keep_ink(img_bgr, s_thr=0.26, v_thr=0.65, thicken=2):
    """Rimuove quadretti preservando l'inchiostro colorato/nero (HSV-driven)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    delta = cmax - cmin + 1e-6
    s = delta / (cmax + 1e-6)
    v = cmax
    keep = (s > float(s_thr)) | (v < float(v_thr))
    if thicken and thicken > 1:
        k = int(thicken)
        keep = cv2.dilate(keep.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
                          1).astype(bool)
    out = np.ones_like(rgb)
    out[keep] = rgb[keep]
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def render_pdf_pages(pdf_path, dpi, out_dir_original):
    """Estrae le pagine del PDF come PNG ad alta qualità (PyMuPDF)."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF non è installato. Installa con: pip install pymupdf") from e
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    out_dir_original.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        p = out_dir_original / f"page_{i:03d}.png"
        pix.save(p.as_posix())
        paths.append(p)
    return paths

# ============================
#  Fase 2: cleaned -> B/W
# ============================

def sauvola_threshold(gray, win=31, k=0.2, R=128):
    gray = gray.astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(win,win), borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(gray*gray, ddepth=-1, ksize=(win,win), borderType=cv2.BORDER_REPLICATE)
    var = mean_sq - mean*mean
    std = cv2.sqrt(np.maximum(var, 0))
    thresh = mean * (1 + k*((std/R)-1))
    bw = (gray > thresh).astype(np.uint8)*255
    return bw

def binarize_image(img_bgr, method="otsu", win=31, k=0.2, remove_specks=18, dilate=0):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if method == "sauvola":
        bw = sauvola_threshold(gray, win=max(3, int(win)//2*2+1), k=float(k))
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = auto_white_background(bw)
    if remove_specks and remove_specks > 0:
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(255-bw, connectivity=8)
        for i in range(1, nb):
            if stats[i, cv2.CC_STAT_AREA] < int(remove_specks):
                bw[labels==i] = 255
    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        bw = cv2.dilate(bw, k, iterations=1)
    return bw

# ============================
#  Fase 3: bw -> strokes (edge‑based vectorization)
# ============================

def to_binary(img_bgr):
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    bw = auto_white_background(bw)
    return bw

def thin_1px(bw_255_white_bg):
    """
    Restituisce skeleton 1‑px (nero=0, bianco=255).
    Preferisce cv2.ximgproc.thinning (Zhang‑Suen), fallback morfologico.
    """
    inv = cv2.bitwise_not(bw_255_white_bg)  # ximgproc vuole fg bianco su sfondo nero
    try:
        import cv2.ximgproc as xi
        skel_fore = xi.thinning(inv, thinningType=xi.THINNING_ZHANGSUEN)
    except Exception:
        # Fallback morfologico
        img = (bw_255_white_bg == 0).astype(np.uint8)  # 1=ink
        skel = np.zeros_like(img, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        while True:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = img - opened
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if not img.any():
                break
        skel_fore = (skel*255).astype(np.uint8)  # foreground white
    return cv2.bitwise_not(skel_fore)  # torna a nero=ink

NEIGH = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

def build_graph(skel):
    ys, xs = np.where(skel == 0)  # nero = tratto
    pixels = list(zip(xs.tolist(), ys.tolist()))
    pix_set = set(pixels)
    degree = {}
    neighbors = {}
    for x, y in pixels:
        nb = []
        for dx, dy in NEIGH:
            nx, ny = x+dx, y+dy
            if (nx, ny) in pix_set:
                nb.append((nx, ny))
        neighbors[(x,y)] = nb
        degree[(x,y)] = len(nb)
    return pix_set, degree, neighbors

def prune_short_branches(pix_set, degree, neighbors, min_len=6):
    """
    Rimuove rametti pendenti (< min_len) ma NON rimuove i dot (grado=0).
    Aiuta a evitare striscioline parallele indesiderate.
    """
    changed = True
    while changed:
        changed = False
        leaves = [p for p in list(pix_set) if degree.get(p,0) == 1]
        for leaf in leaves:
            path = [leaf]
            prev = None
            cur = leaf
            while True:
                nb = [n for n in neighbors.get(cur, []) if n != prev]
                if not nb:
                    break
                nxt = nb[0]
                path.append(nxt)
                if degree.get(nxt, 0) != 2:
                    break
                prev, cur = cur, nxt
                if len(path) >= min_len:
                    break
            if 1 < len(path) < min_len:
                # elimina il ramo (mantieni il nodo di attacco)
                for p in path[:-1]:
                    pix_set.discard(p)
                    for q in neighbors.get(p, []):
                        if q in neighbors and p in neighbors[q]:
                            neighbors[q].remove(p)
                            degree[q] = len(neighbors[q])
                    neighbors.pop(p, None)
                    degree.pop(p, None)
                changed = True
    return pix_set, degree, neighbors

def _edge_key(a, b):
    return (a, b) if a <= b else (b, a)

def trace_chains_edge_based(pix_set, degree, neighbors):
    visited_edges = set()
    chains = []

    # dots (isolati)
    for p in list(pix_set):
        if degree.get(p,0) == 0:
            chains.append([p])

    def walk_from(u, v):
        path = [u, v]
        visited_edges.add(_edge_key(u, v))
        prev, cur = u, v
        while True:
            nxt = None
            for w in neighbors.get(cur, []):
                ek = _edge_key(cur, w)
                if w != prev and ek not in visited_edges:
                    nxt = w
                    break
            if nxt is None or degree.get(cur,0) != 2:
                break
            path.append(nxt)
            visited_edges.add(_edge_key(cur, nxt))
            prev, cur = cur, nxt
        return path

    # estremi/giunzioni (grado != 2)
    for u in pix_set:
        if degree.get(u,0) != 2 and degree.get(u,0) > 0:
            for v in neighbors.get(u, []):
                ek = _edge_key(u, v)
                if ek in visited_edges:
                    continue
                chains.append(walk_from(u, v))

    # cicli puri (tutti grado = 2)
    for u in pix_set:
        if degree.get(u,0) == 2:
            for v in neighbors.get(u, []):
                ek = _edge_key(u, v)
                if ek in visited_edges:
                    continue
                chains.append(walk_from(u, v))

    return [c for c in chains if len(c) > 0]

# RDP per polilinee
def _perp_dist(pt, a, b):
    if a == b:
        return math.hypot(pt[0]-a[0], pt[1]-a[1])
    x, y = pt; x1, y1 = a; x2, y2 = b
    num = abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num/den

def rdp(points, epsilon):
    if len(points) < 3 or epsilon <= 0:
        return points[:]
    dmax = 0.0; idx = 0
    for i in range(1, len(points)-1):
        d = _perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            idx = i; dmax = d
    if dmax > epsilon:
        rec1 = rdp(points[:idx+1], epsilon)
        rec2 = rdp(points[idx:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [points[0], points[-1]]

def vectorize_bw_verbose(img_bgr, epsilon=0.8, prune_branch_len=6, log=lambda *_: None):
    t0 = time.perf_counter()
    bw = to_binary(img_bgr)
    h, w = bw.shape[:2]
    ink = int((bw==0).sum())
    log(f"  • Img {w}x{h} – ink px: {ink:,}")

    # Thinning 1‑px
    t1 = time.perf_counter()
    skel = thin_1px(bw)
    sk_ink = int((skel==0).sum())
    t2 = time.perf_counter()
    log(f"  • Skeleton 1px: {sk_ink:,} px (t={t2-t1:.2f}s)")

    # Grafo
    pix_set, degree, neighbors = build_graph(skel)
    if prune_branch_len and prune_branch_len > 1:
        pix_set, degree, neighbors = prune_short_branches(pix_set, degree, neighbors, prune_branch_len)
        log(f"  • Pruning rami corti < {prune_branch_len}px: OK")

    deg_vals = list(degree.values())
    endpoints = sum(1 for d in deg_vals if d == 1)
    junctions = sum(1 for d in deg_vals if d >= 3)
    log(f"  • Grafo: nodi={len(pix_set):,}, estremi={endpoints:,}, giunzioni={junctions:,}")

    # Catene edge‑based
    chains = trace_chains_edge_based(pix_set, degree, neighbors)
    log(f"  • Catene edge‑based: {len(chains):,}")

    # Polilinee e RDP
    strokes = []
    total_pts_before = 0
    total_pts_after = 0
    for chain in chains:
        pts = [(int(x), int(y)) for (x,y) in chain]
        total_pts_before += len(pts)
        if len(pts) == 1:
            s = {"type": "dot", "points": [pts[0]], "radius_px": 1}
            strokes.append(s)
            total_pts_after += 1
        else:
            simp = rdp(pts, epsilon=float(epsilon)) if epsilon and epsilon > 0 else pts
            strokes.append({"type": "polyline", "points": simp, "closed": (len(pts) > 2 and pts[0] == pts[-1])})
            total_pts_after += len(simp)

    t3 = time.perf_counter()
    red = (1 - (total_pts_after / max(1, total_pts_before))) * 100
    log(f"  • RDP ε={epsilon:.2f}: pts {total_pts_before:,} → {total_pts_after:,} (–{red:.1f}%), t={t3-t2:.2f}s")
    log(f"  • Totale vettorizzazione: {t3-t0:.2f}s")
    return strokes, skel

def preview_overlay(bw_img, strokes, out_path):
    if len(bw_img.shape) == 2:
        base = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)
    else:
        base = bw_img.copy()
    overlay = base.copy()
    for s in strokes:
        if s["type"] == "dot":
            (x,y) = s["points"][0]
            cv2.circle(overlay, (x,y), 2, (0,0,255), -1)
        else:
            pts = np.array(s["points"], dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=False, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, base, 0.3, 0, base)
    cv2.imwrite(str(out_path), base)

# ============================
#  GUI a 3 tab
# ============================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clean Notes — PDF ➜ Immagini ➜ B/N ➜ Tratti (edge‑based)")
        self.geometry("980x700")

        self.base_out_dir = tk.StringVar(value=str(Path.cwd() / "rendered"))

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)

        self.stage1 = Stage1Frame(tabs, self.base_out_dir)
        self.stage2 = Stage2Frame(tabs, self.base_out_dir)
        self.stage3 = Stage3Frame(tabs, self.base_out_dir)

        tabs.add(self.stage1, text="Fase 1: PDF → Immagini")
        tabs.add(self.stage2, text="Fase 2: Cleaned → B/N")
        tabs.add(self.stage3, text="Fase 3: B/N → Tratti")

class BaseFrame(ttk.Frame):
    def log(self, widget, msg):
        widget.insert("end", msg + "\n")
        widget.see("end")
        widget.update_idletasks()

# --------- Fase 1 UI ---------

class Stage1Frame(BaseFrame):
    def __init__(self, master, base_out_var):
        super().__init__(master)
        self.base_out_var = base_out_var

        self.pdf_path = tk.StringVar(value="")
        self.dpi = tk.IntVar(value=600)

        self.enable_antigrid = tk.BooleanVar(value=True)
        self.s_thr = tk.DoubleVar(value=0.26)
        self.v_thr = tk.DoubleVar(value=0.65)
        self.thicken = tk.IntVar(value=2)

        self.build_ui()

    def build_ui(self):
        pad = {'padx': 8, 'pady': 6}
        frm_top = ttk.LabelFrame(self, text="Parametri Fase 1")
        frm_top.pack(fill="x", **pad)

        ttk.Label(frm_top, text="File PDF:").grid(row=0, column=0, sticky="w", **pad)
        self.pdf_out = tk.StringVar(value="")
        ttk.Entry(frm_top, textvariable=self.pdf_out, width=68).grid(row=0, column=1, sticky="we", **pad)
        self.pdf_path = self.pdf_out
        ttk.Button(frm_top, text="Sfoglia…", command=self.pick_pdf).grid(row=0, column=2, **pad)

        ttk.Label(frm_top, text="Cartella base (crea original/cleaned/bw/strokes):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.base_out_var, width=68).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(frm_top, text="Scegli…", command=self.pick_out_dir).grid(row=1, column=2, **pad)

        ttk.Label(frm_top, text="DPI rendering:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.dpi, width=10).grid(row=2, column=1, sticky="w", **pad)

        frm_opts = ttk.LabelFrame(self, text="Anti-griglia (HSV)")
        frm_opts.pack(fill="x", **pad)
        ttk.Checkbutton(frm_opts, text="Rimuovi griglia", variable=self.enable_antigrid).grid(row=0, column=0, sticky="w", **pad)
        ttk.Label(frm_opts, text="S soglia (0–1)").grid(row=0, column=1, sticky="e", **pad)
        ttk.Entry(frm_opts, textvariable=self.s_thr, width=8).grid(row=0, column=2, sticky="w", **pad)
        ttk.Label(frm_opts, text="V soglia (0–1)").grid(row=0, column=3, sticky="e", **pad)
        ttk.Entry(frm_opts, textvariable=self.v_thr, width=8).grid(row=0, column=4, sticky="w", **pad)
        ttk.Label(frm_opts, text="Ispessimento").grid(row=0, column=5, sticky="e", **pad)
        ttk.Entry(frm_opts, textvariable=self.thicken, width=8).grid(row=0, column=6, sticky="w", **pad)

        frm_run = ttk.LabelFrame(self, text="Esecuzione")
        frm_run.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(frm_run, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=8)
        self.logbox = tk.Text(frm_run, height=12, wrap="word")
        self.logbox.pack(fill="both", expand=True, padx=8, pady=(0,8))

        btns = ttk.Frame(frm_run)
        btns.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(btns, text="Anteprima prima pagina", command=self.preview_first).pack(side="left")
        ttk.Button(btns, text="Crea immagini", command=self.start_render).pack(side="left", padx=8)

    def pick_pdf(self):
        p = filedialog.askopenfilename(title="Scegli PDF", filetypes=[("PDF", "*.pdf")])
        if p: self.pdf_path.set(p)

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="Scegli cartella base")
        if d: self.base_out_var.set(d)

    def preview_first(self):
        pdf = self.pdf_path.get().strip()
        base = Path(self.base_out_var.get().strip())
        if not pdf:
            messagebox.showwarning("Attenzione", "Seleziona un PDF.")
            return
        d_orig, d_clean, d_bw, d_strk = ensure_dirs(base)
        try:
            pages = render_pdf_pages(pdf, dpi=int(self.dpi.get()), out_dir_original=d_orig)
            if not pages:
                messagebox.showerror("Errore", "Nessuna pagina renderizzata.")
                return
            p0 = pages[0]
            img = cv2.imread(str(p0), cv2.IMREAD_COLOR)
            if self.enable_antigrid.get():
                img = hsv_antigrid_keep_ink(img, s_thr=float(self.s_thr.get()),
                                                 v_thr=float(self.v_thr.get()),
                                                 thicken=int(self.thicken.get()))
                prev = d_clean / "preview_clean_no_grid.png"
                cv2.imwrite(str(prev), img)
                self.log(self.logbox, f"[ANTEPRIMA] Salvata: {prev}")
                messagebox.showinfo("Anteprima", f"Anteprima salvata:\n{prev}")
            else:
                self.log(self.logbox, "[ANTEPRIMA] Anti-griglia OFF: vedi original/")
        except Exception as e:
            messagebox.showerror("Errore", str(e))

    def start_render(self):
        pdf = self.pdf_path.get().strip()
        base = Path(self.base_out_var.get().strip())
        dpi = int(self.dpi.get())
        if not pdf:
            messagebox.showwarning("Attenzione", "Seleziona un PDF.")
            return
        d_orig, d_clean, d_bw, d_strk = ensure_dirs(base)
        self.progress['value'] = 0
        self.logbox.delete("1.0", "end")
        def worker():
            try:
                originals = render_pdf_pages(pdf, dpi=dpi, out_dir_original=d_orig)
                self.progress.config(maximum=len(originals))
                cleaned = 0
                for i, p in enumerate(originals, start=1):
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None: 
                        self.log(self.logbox, f"[WARN] Impossibile leggere: {p}")
                        continue
                    if self.enable_antigrid.get():
                        img2 = hsv_antigrid_keep_ink(img, s_thr=float(self.s_thr.get()),
                                                          v_thr=float(self.v_thr.get()),
                                                          thicken=int(self.thicken.get()))
                        outp = d_clean / f"{p.stem}_clean.png"
                        cv2.imwrite(str(outp), img2)
                        cleaned += 1
                    self.log(self.logbox, f"[OK] {p.name}")
                    self.progress['value'] = i
                self.log(self.logbox, f"[FATTO] Originali: {len(originals)}  |  Cleaned: {cleaned}")
                messagebox.showinfo("Completato (Fase 1)", "Immagini pronte. Passa alla tab 'Fase 2'.")
            except Exception as e:
                self.log(self.logbox, f"[ERRORE] {e}")
                messagebox.showerror("Errore", str(e))
        threading.Thread(target=worker, daemon=True).start()

# --------- Fase 2 UI ---------

class Stage2Frame(BaseFrame):
    def __init__(self, master, base_out_var):
        super().__init__(master)
        self.base_out_var = base_out_var
        self.method = tk.StringVar(value="otsu")
        self.win = tk.IntVar(value=31)
        self.k = tk.DoubleVar(value=0.2)
        self.remove_specks = tk.IntVar(value=18)
        self.dilate = tk.IntVar(value=0)
        self.build_ui()

    def build_ui(self):
        pad = {'padx': 8, 'pady': 6}
        frm_top = ttk.LabelFrame(self, text="Parametri Fase 2 (da 'cleaned' a B/N)")
        frm_top.pack(fill="x", **pad)

        ttk.Label(frm_top, text="Cartella base:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.base_out_var, width=68).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm_top, text="Scegli…", command=self.pick_out_dir).grid(row=0, column=2, **pad)

        ttk.Label(frm_top, text="Metodo:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm_top, text="Otsu", variable=self.method, value="otsu").grid(row=1, column=1, sticky="w", **pad)
        ttk.Radiobutton(frm_top, text="Sauvola", variable=self.method, value="sauvola").grid(row=1, column=2, sticky="w", **pad)

        ttk.Label(frm_top, text="Finestra (Sauvola):").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.win, width=8).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(frm_top, text="k (Sauvola):").grid(row=2, column=2, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.k, width=8).grid(row=2, column=3, sticky="w", **pad)

        ttk.Label(frm_top, text="Rimuovi puntinature <=").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.remove_specks, width=8).grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(frm_top, text="Dilatazione px:").grid(row=3, column=2, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.dilate, width=8).grid(row=3, column=3, sticky="w", **pad)

        frm_run = ttk.LabelFrame(self, text="Esecuzione")
        frm_run.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(frm_run, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=8)
        self.logbox = tk.Text(frm_run, height=12, wrap="word")
        self.logbox.pack(fill="both", expand=True, padx=8, pady=(0,8))

        btns = ttk.Frame(frm_run)
        btns.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(btns, text="Anteprima (prima immagine)", command=self.preview).pack(side="left")
        ttk.Button(btns, text="Converti tutto in B/N", command=self.convert_all).pack(side="left", padx=8)

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="Scegli cartella base")
        if d: self.base_out_var.set(d)

    def preview(self):
        base = Path(self.base_out_var.get().strip())
        d_orig, d_clean, d_bw, d_strk = ensure_dirs(base)
        imgs = list_images(d_clean)
        if not imgs:
            messagebox.showwarning("Attenzione", "Nessuna immagine in 'cleaned'. Esegui prima la Fase 1.")
            return
        img = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
        bw = binarize_image(img, method=self.method.get(), win=int(self.win.get()), k=float(self.k.get()),
                            remove_specks=int(self.remove_specks.get()), dilate=int(self.dilate.get()))
        prev = d_bw / "preview_bw.png"
        cv2.imwrite(str(prev), bw)
        self.log(self.logbox, f"[ANTEPRIMA] Salvata: {prev}")
        messagebox.showinfo("Anteprima", f"Anteprima salvata:\n{prev}")

    def convert_all(self):
        base = Path(self.base_out_var.get().strip())
        d_orig, d_clean, d_bw, d_strk = ensure_dirs(base)
        imgs = list_images(d_clean)
        if not imgs:
            messagebox.showwarning("Attenzione", "Nessuna immagine in 'cleaned'.")
            return
        self.progress['value'] = 0
        self.logbox.delete("1.0", "end")
        def worker():
            try:
                self.progress.config(maximum=len(imgs))
                for i, p in enumerate(imgs, start=1):
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    bw = binarize_image(img, method=self.method.get(), win=int(self.win.get()), k=float(self.k.get()),
                                        remove_specks=int(self.remove_specks.get()), dilate=int(self.dilate.get()))
                    outp = d_bw / f"{p.stem}_bw.png"
                    cv2.imwrite(str(outp), bw)
                    self.log(self.logbox, f"[OK] {outp.name}")
                    self.progress['value'] = i
                self.log(self.logbox, f"[FATTO] Convertite: {len(imgs)} (cartella 'bw')")
                messagebox.showinfo("Completato (Fase 2)", "Conversione in B/N terminata.")
            except Exception as e:
                self.log(self.logbox, f"[ERRORE] {e}")
                messagebox.showerror("Errore", str(e))
        threading.Thread(target=worker, daemon=True).start()

# --------- Fase 3 UI ---------

class Stage3Frame(BaseFrame):
    def __init__(self, master, base_out_var):
        super().__init__(master)
        self.base_out_var = base_out_var
        self.epsilon = tk.DoubleVar(value=0.8)     # RDP epsilon (px)
        self.prune_len = tk.IntVar(value=6)        # pruning rametti (px)
        self.preview_n = tk.IntVar(value=1)        # quante immagini in anteprima
        self.build_ui()

    def build_ui(self):
        pad = {'padx': 8, 'pady': 6}
        frm_top = ttk.LabelFrame(self, text="Parametri Fase 3 (da 'bw' a 'strokes')")
        frm_top.pack(fill="x", **pad)

        ttk.Label(frm_top, text="Cartella base:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.base_out_var, width=68).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm_top, text="Scegli…", command=self.pick_out_dir).grid(row=0, column=2, **pad)

        ttk.Label(frm_top, text="RDP epsilon (px):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.epsilon, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(frm_top, text="Pruning rametti < (px):").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.prune_len, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(frm_top, text="Anteprime (n pagine):").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.preview_n, width=8).grid(row=2, column=1, sticky="w", **pad)

        frm_run = ttk.LabelFrame(self, text="Esecuzione (log dettagliato)")
        frm_run.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(frm_run, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=8)
        self.logbox = tk.Text(frm_run, height=14, wrap="word")
        self.logbox.pack(fill="both", expand=True, padx=8, pady=(0,8))

        btns = ttk.Frame(frm_run)
        btns.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(btns, text="Anteprima primi N", command=self.preview_some).pack(side="left")
        ttk.Button(btns, text="Vettorizza tutto", command=self.vectorize_all).pack(side="left", padx=8)

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="Scegli cartella base")
        if d: self.base_out_var.set(d)

    def _log(self, msg):
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")
        self.logbox.update_idletasks()

    def _vectorize_single(self, bw_path: Path, d_strk: Path):
        self._log(f"[FILE] {bw_path.name}")
        t0 = time.perf_counter()
        img = cv2.imread(str(bw_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Impossibile leggere {bw_path}")
        h, w = img.shape[:2]
        self._log(f"  • Dimensioni: {w}x{h}")
        strokes, skel = vectorize_bw_verbose(img,
                                             epsilon=float(self.epsilon.get()),
                                             prune_branch_len=int(self.prune_len.get()),
                                             log=self._log)

        # salva JSON compatto
        data = {
            "version": "1.0",
            "image_size": {"width": int(w), "height": int(h)},
            "dpi": None,
            "source": str(bw_path.name),
            "strokes": strokes
        }
        out_json = d_strk / f"{bw_path.stem}_strokes.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))  # compatto

        # salva overlay di controllo
        bw = to_binary(img)
        out_prev = d_strk / f"{bw_path.stem}_strokes_preview.png"
        preview_overlay(bw, strokes, out_prev)

        self._log(f"  • Salvati: {out_json.name}, {out_prev.name} (t={time.perf_counter()-t0:.2f}s)")
        return out_json, out_prev

    def preview_some(self):
        base = Path(self.base_out_var.get().strip())
        d_orig, d_clean, d_bw, d_strk = ensure_dirs(base)
        imgs = list_images(d_bw)
        if not imgs:
            messagebox.showwarning("Attenzione", "Nessuna immagine in 'bw'. Esegui prima la Fase 2.")
            return
        n = max(1, int(self.preview_n.get()))
        sel = imgs[:n]
        self.progress['value'] = 0
        self.progress.config(maximum=len(sel))
        self.logbox.delete("1.0", "end")
        start = time.perf_counter()
        def worker():
            try:
                for i, p in enumerate(sel, start=1):
                    self._vectorize_single(p, d_strk)
                    self._log(f"[OK] Anteprima {i}/{len(sel)}")
                    self.progress['value'] = i
                self._log(f"[DONE] Anteprime: {len(sel)} pagine in {time.perf_counter()-start:.2f}s")
                messagebox.showinfo("Anteprima vettoriale", f"Creati {len(sel)} file di anteprima in 'strokes'.")
            except Exception as e:
                self._log(f"[ERRORE] {e}")
                messagebox.showerror("Errore", str(e))
        threading.Thread(target=worker, daemon=True).start()

    def vectorize_all(self):
        base = Path(self.base_out_var.get().strip())
        d_orig, d_clean, d_bw, d_strk = ensure_dirs(base)
        imgs = list_images(d_bw)
        if not imgs:
            messagebox.showwarning("Attenzione", "Nessuna immagine in 'bw'.")
            return
        self.progress['value'] = 0
        self.progress.config(maximum=len(imgs))
        self.logbox.delete("1.0", "end")
        start = time.perf_counter()
        def worker():
            try:
                for i, p in enumerate(imgs, start=1):
                    self._vectorize_single(p, d_strk)
                    self._log(f"[OK] Pagina {i}/{len(imgs)}")
                    self.progress['value'] = i
                self._log(f"[DONE] Vettorizzate: {len(imgs)} pagine in {time.perf_counter()-start:.2f}s")
                messagebox.showinfo("Completato (Fase 3)", f"Creati {len(imgs)} JSON di tratti in 'strokes'.")
            except Exception as e:
                self._log(f"[ERRORE] {e}")
                messagebox.showerror("Errore", str(e))
        threading.Thread(target=worker, daemon=True).start()

# ============================

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
