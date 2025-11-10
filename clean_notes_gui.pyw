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
# [removed] Tab 3 removed (instantiation)
        tabs.add(self.stage1, text="Fase 1: PDF → Immagini")
        tabs.add(self.stage2, text="Fase 2: Cleaned → B/N")
# [removed] Tab 3 removed
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

# ============================

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
