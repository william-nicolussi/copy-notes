#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Clean Notes GUI — PDF ➜ Images ➜ B/W
Fasi implementate in modo robusto e minimale (niente placeholder o '...'):

- Fase 1: PDF -> immagini (original/cleaned) con anti‑griglia (HSV) opzionale
- Fase 2: cleaned -> bianco/nero (bw) con Otsu/Sauvola

Dipendenze principali (installare in anticipo):
  pip install pymupdf opencv-python numpy

Note:
- Rimossi import inutili (json, math, time).
- Sistemato shadowing di variabili (k_sauvola vs kernel).
- Nessun ellissi/placeholder: il file è eseguibile così com'è.
"""

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
    base_dir = Path(base_dir)
    d_orig = base_dir / "original"
    d_clean = base_dir / "cleaned"
    d_bw = base_dir / "bw"
    d_strk = base_dir / "strokes"  # riservato per step futuri
    for d in (d_orig, d_clean, d_bw, d_strk):
        d.mkdir(parents=True, exist_ok=True)
    return d_orig, d_clean, d_bw, d_strk


def list_images(folder: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted([p for p in Path(folder).glob("*") if p.suffix.lower() in exts])


def auto_white_background(bw_img: np.ndarray) -> np.ndarray:
    """Se il bordo medio è scuro, inverte per avere sfondo bianco e segno nero."""
    border = np.concatenate([bw_img[0, :], bw_img[-1, :], bw_img[:, 0], bw_img[:, -1]])
    if border.mean() < 127:
        bw_img = cv2.bitwise_not(bw_img)
    return bw_img


# ============================
#  Fase 1: PDF -> original/cleaned
# ============================

def hsv_antigrid_keep_ink(img_bgr: np.ndarray, s_thr=0.26, v_thr=0.65, thicken=2) -> np.ndarray:
    """
    Rimuove quadretti preservando l'inchiostro colorato/nero (HSV-like su RGB normalizzato).
    keep = (saturazione alta) OR (valore scuro)
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    delta = cmax - cmin
    s = delta / (cmax + 1e-6)
    v = cmax

    keep = (s > float(s_thr)) | (v < float(v_thr))
    if thicken and thicken > 1:
        k_size = int(thicken)
        keep = cv2.dilate(
            keep.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size)),
            iterations=1,
        ).astype(bool)

    out = img_bgr.copy()
    # I pixel NON keep (griglia/pagina) vengono schiariti/spinti verso bianco
    mask = ~keep
    out[mask] = (out[mask].astype(np.float32) * 0.2 + 255 * 0.8).astype(np.uint8)
    return out


def render_pdf_pages(pdf_path: str, dpi: int, out_dir_original: Path):
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
        pix = page.get_pixmap(matrix=mat, alpha=False)  # niente alpha
        p = out_dir_original / f"page_{i:03d}.png"
        pix.save(p.as_posix())
        paths.append(p)
    return paths


# ============================
#  Fase 2: cleaned -> B/W
# ============================

def sauvola_threshold(gray: np.ndarray, win=31, k_sauvola=0.2, R=128):
    """Calcolo Sauvola: soglia locale per binarizzazione di documenti."""
    gray = gray.astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(win, win), borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(win, win), borderType=cv2.BORDER_REPLICATE)
    var = mean_sq - mean * mean
    std = cv2.sqrt(np.maximum(var, 0))
    thresh = mean * (1 + k_sauvola * ((std / R) - 1))
    bw = (gray > thresh).astype(np.uint8) * 255
    return bw


def binarize_image(img_bgr: np.ndarray, method="otsu", win=31, k_sauvola=0.2,
                   remove_specks=0, dilate_px=0) -> np.ndarray:
    """Binarizza immagine con Otsu o Sauvola; opzionale rimozione puntinature e dilatazione."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if method == "sauvola":
        bw = sauvola_threshold(gray, win=win, k_sauvola=k_sauvola)
    else:
        thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if remove_specks and remove_specks > 0:
        nb = remove_specks
        bw = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (nb, nb)),
            iterations=1
        )

    if dilate_px and dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
        bw = cv2.dilate(bw, kernel, iterations=1)

    bw = auto_white_background(bw)
    return bw


# ============================
#  UI
# ============================

class BaseFrame(ttk.Frame):
    def log(self, widget, msg: str):
        widget.insert("end", msg + "\n")
        widget.see("end")
        widget.update_idletasks()


# --------- Fase 1 UI ---------

class Stage1Frame(BaseFrame):
    def __init__(self, master, base_out_var: tk.StringVar):
        super().__init__(master)
        self.base_out_var = base_out_var

        self.pdf_path = tk.StringVar(value="")
        self.dpi = tk.IntVar(value=600)

        self.enable_antigrid = tk.BooleanVar(value=True)
        self.s_thr = tk.DoubleVar(value=0.26)
        self.v_thr = tk.DoubleVar(value=0.65)
        self.thicken = tk.IntVar(value=2)

        pad = dict(padx=8, pady=6)

        # File choose
        frm_pdf = ttk.Frame(self)
        frm_pdf.pack(fill="x", **pad)
        ttk.Label(frm_pdf, text="PDF:").pack(side="left")
        ttk.Entry(frm_pdf, textvariable=self.pdf_path, width=60).pack(side="left", padx=6)
        ttk.Button(frm_pdf, text="Sfoglia…", command=self.choose_pdf).pack(side="left")

        # Options
        frm_opts = ttk.LabelFrame(self, text="Opzioni rendering")
        frm_opts.pack(fill="x", **pad)
        ttk.Label(frm_opts, text="DPI:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_opts, textvariable=self.dpi, width=8).grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(frm_opts, text="Anti-griglia (HSV)", variable=self.enable_antigrid).grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(frm_opts, text="S soglia:").grid(row=1, column=1, sticky="w", **pad)
        ttk.Entry(frm_opts, textvariable=self.s_thr, width=8).grid(row=1, column=2, sticky="w", **pad)
        ttk.Label(frm_opts, text="V soglia:").grid(row=1, column=3, sticky="w", **pad)
        ttk.Entry(frm_opts, textvariable=self.v_thr, width=8).grid(row=1, column=4, sticky="w", **pad)
        ttk.Label(frm_opts, text="Thicken px:").grid(row=1, column=5, sticky="w", **pad)
        ttk.Entry(frm_opts, textvariable=self.thicken, width=8).grid(row=1, column=6, sticky="w", **pad)

        # Run
        frm_run = ttk.LabelFrame(self, text="Esecuzione")
        frm_run.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(frm_run, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=8)
        self.logbox = tk.Text(frm_run, height=10)
        self.logbox.pack(fill="both", expand=True, padx=8, pady=6)

        frm_btns = ttk.Frame(frm_run)
        frm_btns.pack(fill="x", pady=6)
        ttk.Button(frm_btns, text="Anteprima (pag.1)", command=self.preview_first).pack(side="left", padx=4)
        ttk.Button(frm_btns, text="Render tutte le pagine", command=self.start_render).pack(side="left", padx=4)

    def choose_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if path:
            self.pdf_path.set(path)

    def preview_first(self):
        pdf = self.pdf_path.get().strip()
        if not pdf:
            messagebox.showwarning("Attenzione", "Seleziona un PDF.")
            return
        base = Path(self.base_out_var.get().strip())
        d_orig, d_clean, d_bw, d_strk = ensure_dirs(base)
        try:
            pages = render_pdf_pages(pdf, dpi=int(self.dpi.get()), out_dir_original=d_orig)
            if not pages:
                messagebox.showerror("Errore", "Nessuna pagina renderizzata.")
                return
            p0 = pages[0]
            img = cv2.imread(str(p0), cv2.IMREAD_COLOR)
            if self.enable_antigrid.get():
                img = hsv_antigrid_keep_ink(
                    img, s_thr=float(self.s_thr.get()),
                    v_thr=float(self.v_thr.get()),
                    thicken=int(self.thicken.get())
                )
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

        def worker():
            try:
                self.log(self.logbox, f"[RENDER] Estraggo pagine da {pdf} @ {dpi} DPI …")
                pages = render_pdf_pages(pdf, dpi=dpi, out_dir_original=d_orig)
                n = len(pages)
                if n == 0:
                    raise RuntimeError("Nessuna pagina nel PDF.")
                for i, p in enumerate(pages, start=1):
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    out_img = img
                    if self.enable_antigrid.get():
                        out_img = hsv_antigrid_keep_ink(
                            img,
                            s_thr=float(self.s_thr.get()),
                            v_thr=float(self.v_thr.get()),
                            thicken=int(self.thicken.get())
                        )
                    out_p = d_clean / p.name
                    cv2.imwrite(str(out_p), out_img)
                    self.progress['value'] = int(i * 100 / n)
                    self.log(self.logbox, f"[RENDER] {i}/{n} -> {out_p}")
                messagebox.showinfo("Completato (Fase 1)", "Render completato.")
            except Exception as e:
                self.log(self.logbox, f"[ERRORE] {e}")
                messagebox.showerror("Errore", str(e))

        threading.Thread(target=worker, daemon=True).start()


# --------- Fase 2 UI ---------

class Stage2Frame(BaseFrame):
    def __init__(self, master, base_out_var: tk.StringVar):
        super().__init__(master)
        self.base_out_var = base_out_var

        self.method = tk.StringVar(value="otsu")  # 'otsu' | 'sauvola'
        self.win = tk.IntVar(value=31)
        self.k = tk.DoubleVar(value=0.2)  # k di Sauvola
        self.remove_specks = tk.IntVar(value=0)
        self.dilate = tk.IntVar(value=0)

        pad = dict(padx=8, pady=6)

        frm_top = ttk.LabelFrame(self, text="Parametri")
        frm_top.pack(fill="x", **pad)

        ttk.Label(frm_top, text="Metodo:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm_top, text="Otsu", variable=self.method, value="otsu").grid(row=0, column=1, sticky="w", **pad)
        ttk.Radiobutton(frm_top, text="Sauvola", variable=self.method, value="sauvola").grid(row=0, column=2, sticky="w", **pad)

        ttk.Label(frm_top, text="Finestra (Sauvola):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.win, width=8).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(frm_top, text="k (Sauvola):").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.k, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(frm_top, text="Rimuovi puntinature ≤").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.remove_specks, width=8).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(frm_top, text="Dilatazione px:").grid(row=2, column=2, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.dilate, width=8).grid(row=2, column=3, sticky="w", **pad)

        frm_run = ttk.LabelFrame(self, text="Esecuzione")
        frm_run.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(frm_run, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=8)
        self.logbox = tk.Text(frm_run, height=10)
        self.logbox.pack(fill="both", expand=True, padx=8, pady=6)

        frm_btns = ttk.Frame(frm_run)
        frm_btns.pack(fill="x", pady=6)
        ttk.Button(frm_btns, text="Anteprima (prima immagine)", command=self.preview_first).pack(side="left", padx=4)
        ttk.Button(frm_btns, text="Converti tutte (cleaned → bw)", command=self.convert_all).pack(side="left", padx=4)

    def preview_first(self):
        base = Path(self.base_out_var.get().strip())
        d_orig, d_clean, d_bw, d_strk = ensure_dirs(base)
        imgs = list_images(d_clean)
        if not imgs:
            messagebox.showwarning("Attenzione", "Nessuna immagine in 'cleaned'.")
            return
        img = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
        bw = binarize_image(
            img,
            method=self.method.get(),
            win=int(self.win.get()),
            k_sauvola=float(self.k.get()),
            remove_specks=int(self.remove_specks.get()),
            dilate_px=int(self.dilate.get())
        )
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

        def worker():
            try:
                n = len(imgs)
                for i, p in enumerate(imgs, start=1):
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    bw = binarize_image(
                        img,
                        method=self.method.get(),
                        win=int(self.win.get()),
                        k_sauvola=float(self.k.get()),
                        remove_specks=int(self.remove_specks.get()),
                        dilate_px=int(self.dilate.get())
                    )
                    out_p = d_bw / p.name
                    cv2.imwrite(str(out_p), bw)
                    self.progress['value'] = int(i * 100 / n)
                    self.log(self.logbox, f"[BW] {i}/{n} -> {out_p}")
                messagebox.showinfo("Completato (Fase 2)", "Conversione in B/N terminata.")
            except Exception as e:
                self.log(self.logbox, f"[ERRORE] {e}")
                messagebox.showerror("Errore", str(e))

        threading.Thread(target=worker, daemon=True).start()


# --------- App container ---------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clean Notes GUI — PDF → Cleaned → B/W")
        self.geometry("860x640")

        # cartella di output base
        self.base_out_dir = tk.StringVar(value=str(Path.cwd() / "rendered"))

        # chooser per cartella base
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="Cartella output:").pack(side="left")
        ttk.Entry(top, textvariable=self.base_out_dir, width=60).pack(side="left", padx=6)
        ttk.Button(top, text="Scegli cartella…", command=self.choose_out_dir).pack(side="left")

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)

        self.stage1 = Stage1Frame(tabs, self.base_out_dir)
        self.stage2 = Stage2Frame(tabs, self.base_out_dir)
        tabs.add(self.stage1, text="Fase 1: PDF → Immagini")
        tabs.add(self.stage2, text="Fase 2: Cleaned → B/N")

    def choose_out_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.base_out_dir.set(d)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
