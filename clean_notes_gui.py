# ==== PATCH START: Verbose Vectorization Tab (Stage 3) ====
# Add (or keep) these imports near the top of your file:
import time
import json
import math
import numpy as np
import cv2
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
import tkinter as tk

# ---------- Helper functions used by Stage 3 (safe to reuse existing ones) ----------

def auto_white_background(bw_img):
    border = np.concatenate([bw_img[0,:], bw_img[-1,:], bw_img[:,0], bw_img[:,-1]])
    if border.mean() < 127:
        bw_img = cv2.bitwise_not(bw_img)
    return bw_img

def list_images(folder: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])

def ensure_dirs(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    d_orig = base_dir / "original"
    d_clean = base_dir / "cleaned"
    d_bw = base_dir / "bw"
    d_strk = base_dir / "strokes"
    for d in (d_orig, d_clean, d_bw, d_strk):
        d.mkdir(exist_ok=True)
    return d_orig, d_clean, d_bw, d_strk

def to_binary(img_bgr):
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    bw = auto_white_background(bw)
    return bw

def morphological_skeleton(bw):
    # Expect bw as 0/255; treat black(0) as ink
    img = (bw == 0).astype(np.uint8)
    skel = np.zeros_like(img, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # Iterate until full erosion
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = img - opened
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if not img.any():
            break
    return (skel*255).astype(np.uint8)

def _perp_dist(pt, a, b):
    if a == b:
        return math.hypot(pt[0]-a[0], pt[1]-a[1])
    x, y = pt; x1, y1 = a; x2, y2 = b
    num = abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num/den

def rdp(points, epsilon):
    if len(points) < 3:
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

NEIGH = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

def build_graph(skel):
    ys, xs = np.where(skel == 0)  # black pixels
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

def trace_paths(pix_set, degree, neighbors):
    visited = set()
    strokes = []

    def walk(start, prev=None):
        path = [start]
        cur = start
        while True:
            nb = [p for p in neighbors[cur] if p != prev]
            nxt_candidates = [p for p in nb if p not in visited]
            if degree[cur] != 2:
                break
            nxt = nxt_candidates[0] if nxt_candidates else (nb[0] if nb else None)
            if nxt is None:
                break
            prev, cur = cur, nxt
            if cur in path:
                break
            path.append(cur)
            visited.add(cur)
        return path

    endpoints = [p for p in pix_set if degree[p] == 1]
    for ep in endpoints:
        if ep in visited:
            continue
        visited.add(ep)
        path = walk(ep, prev=None)
        strokes.append(path)

    for p in pix_set:
        if p in visited:
            continue
        if degree[p] != 2:
            for nb in neighbors[p]:
                if nb not in visited:
                    visited.add(p)
                    path = [p] + walk(nb, prev=p)
                    strokes.append(path)
        else:
            path = [p]
            visited.add(p)
            prev = None
            cur = p
            while True:
                nb = [q for q in neighbors[cur] if q != prev]
                nxt = None
                for q in nb:
                    if q not in visited:
                        nxt = q; break
                if nxt is None:
                    break
                prev, cur = cur, nxt
                visited.add(cur)
                path.append(cur)
                if cur == p:
                    break
            strokes.append(path)

    strokes = [s for s in strokes if len(s) > 0]
    return strokes

def vectorize_bw_verbose(bw_img, epsilon=0.8, log=lambda *_: None):
    t0 = time.perf_counter()
    bw = to_binary(bw_img)
    h, w = bw.shape[:2]
    ink = int((bw==0).sum())
    log(f"  • Img {w}x{h} – pixel inchiostro: {ink:,}")

    t1 = time.perf_counter()
    skel = morphological_skeleton(bw)
    sk_ink = int((skel==0).sum())
    t2 = time.perf_counter()
    log(f"  • Scheletro: {sk_ink:,} pixel (t={t2-t1:.2f}s)")

    pix_set, degree, neighbors = build_graph(skel)
    deg_vals = list(degree.values())
    endpoints = sum(1 for d in deg_vals if d == 1)
    junctions = sum(1 for d in deg_vals if d >= 3)
    log(f"  • Grafo: nodi={len(pix_set):,}, estremi={endpoints:,}, giunzioni={junctions:,}")

    raw_strokes = trace_paths(pix_set, degree, neighbors)
    log(f"  • Cammini grezzi: {len(raw_strokes):,}")

    strokes = []
    total_pts_before = 0
    total_pts_after = 0
    for path in raw_strokes:
        pts = [(int(x), int(y)) for (x,y) in path]
        total_pts_before += len(pts)
        if len(pts) == 1:
            s = {"type": "dot", "points": [pts[0]], "radius_px": 1}
            strokes.append(s)
            total_pts_after += 1
        else:
            simp = rdp(pts, epsilon=float(epsilon)) if epsilon and epsilon > 0 else pts
            strokes.append({"type": "polyline", "points": simp, "closed": (len(pts) > 3 and pts[0] == pts[-1])})
            total_pts_after += len(simp)

    t3 = time.perf_counter()
    red = (1 - (total_pts_after / max(1, total_pts_before))) * 100
    log(f"  • RDP ε={epsilon:.2f}: punti {total_pts_before:,} → {total_pts_after:,} (–{red:.1f}%), t={t3-t2:.2f}s")
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

# ---------- REPLACE your current Stage3Frame with this verbose version ----------

class Stage3Frame(ttk.Frame):
    def __init__(self, master, base_out_var):
        super().__init__(master)
        self.base_out_var = base_out_var
        self.epsilon = tk.DoubleVar(value=0.8)
        self.preview_n = tk.IntVar(value=1)
        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}
        frm_top = ttk.LabelFrame(self, text="Parametri Fase 3 (da 'bw' a 'strokes')")
        frm_top.pack(fill="x", **pad)

        ttk.Label(frm_top, text="Cartella base:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.base_out_var, width=68).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm_top, text="Scegli…", command=self._pick_out_dir).grid(row=0, column=2, **pad)

        ttk.Label(frm_top, text="RDP epsilon (px):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.epsilon, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(frm_top, text="Anteprime (n pagine):").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.preview_n, width=8).grid(row=1, column=3, sticky="w", **pad)

        frm_run = ttk.LabelFrame(self, text="Esecuzione (log dettagliato)")
        frm_run.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(frm_run, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=8)
        self.logbox = tk.Text(frm_run, height=14, wrap="word")
        self.logbox.pack(fill="both", expand=True, padx=8, pady=(0,8))

        btns = ttk.Frame(frm_run)
        btns.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(btns, text="Anteprima primi N", command=self._preview_some).pack(side="left")
        ttk.Button(btns, text="Vettorizza tutto", command=self._vectorize_all).pack(side="left", padx=8)

    def _pick_out_dir(self):
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
        strokes, skel = vectorize_bw_verbose(img, epsilon=float(self.epsilon.get()), log=self._log)

        data = {
            "version": "1.0",
            "image_size": {"width": int(w), "height": int(h)},
            "dpi": None,
            "source": str(bw_path.name),
            "strokes": strokes
        }
        out_json = d_strk / f"{bw_path.stem}_strokes.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        out_prev = d_strk / f"{bw_path.stem}_strokes_preview.png"
        preview_overlay(to_binary(img), strokes, out_prev)

        self._log(f"  • Salvati: {out_json.name}, {out_prev.name} (t={time.perf_counter()-t0:.2f}s)")
        return out_json, out_prev

    def _preview_some(self):
        base = Path(self.base_out_var.get().strip())
        _, _, d_bw, d_strk = ensure_dirs(base)
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

    def _vectorize_all(self):
        base = Path(self.base_out_var.get().strip())
        _, _, d_bw, d_strk = ensure_dirs(base)
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

# ==== PATCH END ====
