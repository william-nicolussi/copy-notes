# -*- coding: utf-8 -*-
"""
write_notes_gui.pyw (Tkinter)
Fase 1: Caricamento JSON (uno per pagina) + Calibrazione canvas Samsung Notes a 2 click (TL, BR)

Dipendenze: solo standard library (tkinter, json, glob, os, sys).
Nessun PyQt5 richiesto.

Uso:
  - Esegui con Python (doppio click o `python write_notes_gui.pyw`).
  - Seleziona la cartella con i JSON.
  - Seleziona un file dalla lista per vedere l'anteprima vettoriale.
  - Premi "Calibra (2 click)" e, sull'overlay full-screen semitrasparente, clicca:
      1) angolo Alto-Sinistra del foglio in Samsung Notes
      2) angolo Basso-Destra del foglio
  - Salva la calibrazione su file JSON.

Nota: l'anteprima disegna al massimo i primi N stroke per performance.
"""

import os
import sys
import json
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -----------------------------
# Modello dati base
# -----------------------------

@dataclass
class Stroke:
    points: List[Tuple[float, float]]
    closed: bool = False

@dataclass
class PageData:
    width: int
    height: int
    strokes: List[Stroke]
    path: str

# -----------------------------
# Overlay di calibrazione full-screen (semi-trasparente)
# -----------------------------

class CalibOverlay(tk.Toplevel):
    """Finestra full-screen semi-trasparente per catturare 2 click (TL, BR)."""
    def __init__(self, master, on_done):
        super().__init__(master)
        self.on_done = on_done  # callback (tl_xy, br_xy)
        self.points: List[Tuple[int, int]] = []
        self._last_click_ms: int = 0  # per evitare doppio-bind sullo stesso click

        # Rendi la finestra un overlay full-screen
        self.overrideredirect(True)
        try:
            self.attributes('-alpha', 0.30)
        except Exception:
            pass
        self.attributes('-topmost', True)
        try:
            self.attributes('-fullscreen', True)
        except Exception:
            self.state('zoomed')

        # Canvas per istruzioni e feedback
        self.canvas = tk.Canvas(self, bg='black')
        self.canvas.pack(fill='both', expand=True)

        # Bind SOLO sul canvas (evita doppio evento su toplevel+canvas)
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<Escape>', self._on_escape)
        self.bind('<Escape>', self._on_escape)

        # Porta in primo piano e cattura input
        self.update_idletasks()
        self.lift()
        self.focus_force()
        try:
            self.grab_set_global()
        except Exception:
            self.grab_set()

        self._draw_instructions()

    def _draw_instructions(self):
        self.canvas.delete('all')
        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        msg = (
            "Calibrazione: clicca 1) Top-Left e 2) Bottom-Right del canvas Samsung Notes"
            "ESC per annullare"
        )
        self.canvas.create_text(w//2, 40, text=msg, fill='white', font=('Segoe UI', 16, 'bold'))
        for i, (x, y) in enumerate(self.points, start=1):
            self.canvas.create_oval(x-8, y-8, x+8, y+8, outline='lime', width=2)
            self.canvas.create_text(x+14, y-14, text=f'P{i}', fill='lime', font=('Segoe UI', 12, 'bold'))
        if len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            self.canvas.create_rectangle(x1, y1, x2, y2, outline='lime', width=2)

    def _on_escape(self, _):
        self.points.clear()
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _on_click(self, event):
        # Debounce: ignora doppio firing dello stesso click entro 150 ms
        now = int(self.tk.call('clock', 'milliseconds'))
        if now - self._last_click_ms < 150:
            return
        self._last_click_ms = now

        # Coordinate globali affidabili dal Motion/Click event
        x = event.x_root
        y = event.y_root

        # Se prima e seconda posizione sono troppo vicine (<3 px), ignora come duplicato
        if self.points:
            x0, y0 = self.points[-1]
            if abs(x - x0) < 3 and abs(y - y0) < 3:
                return

        self.points.append((x, y))
        self._draw_instructions()
        if len(self.points) == 2:
            tl, br = self.points
            # Rilascia la grab prima di chiudere
            try:
                self.grab_release()
            except Exception:
                pass
            self.after(50, lambda: (self.destroy(), self.on_done(tl, br)))

    def _draw_instructions(self):
        self.canvas.delete('all')
        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        # Testo istruzioni
        msg = (
            "Calibrazione: clicca 1) Top-Left e 2) Bottom-Right del canvas Samsung Notes\n"
            "ESC per annullare"
        )
        self.canvas.create_text(w//2, 40, text=msg, fill='white', font=('Segoe UI', 16, 'bold'))
        # Disegna i punti già catturati
        for i, (x, y) in enumerate(self.points, start=1):
            self.canvas.create_oval(x-8, y-8, x+8, y+8, outline='lime', width=2)
            self.canvas.create_text(x+14, y-14, text=f'P{i}', fill='lime', font=('Segoe UI', 12, 'bold'))
        if len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            self.canvas.create_rectangle(x1, y1, x2, y2, outline='lime', width=2)

    def _on_escape(self, _):
        self.points.clear()
        self.destroy()

    def _on_click(self, event):
        # Usa coordinate globali del puntatore per essere indipendente dal canvas
        x = self.winfo_pointerx()
        y = self.winfo_pointery()
        self.points.append((x, y))
        self._draw_instructions()
        if len(self.points) == 2:
            tl, br = self.points
            self.after(50, lambda: (self.destroy(), self.on_done(tl, br)))

# -----------------------------
# Finestra principale: Fase 1
# -----------------------------

class App(tk.Tk):
    PREVIEW_MAX_STROKES = 1500  # nessun limite di default (era 1500)

    def __init__(self):
        super().__init__()
        self.title('Samsung Notes Writer – Fase 1 (JSON & Calibrazione)')
        self.geometry('1100x700')

        # Stato
        self.folder_path: Optional[str] = None
        self.pages: List[PageData] = []
        self.current_index: int = -1
        self.screen_tl: Optional[Tuple[int, int]] = None
        self.screen_br: Optional[Tuple[int, int]] = None

        # Layout principale con Notebook (tabs) per future fasi
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True)

        self.phase1 = ttk.Frame(self.nb)
        self.nb.add(self.phase1, text='Fase 1 – JSON & Calibrazione')

        self._build_phase1()

    # -------------------------
    # UI Fase 1
    # -------------------------
    def _build_phase1(self):
        left = ttk.Frame(self.phase1)
        left.pack(side='left', fill='y', padx=8, pady=8)

        btn_select = ttk.Button(left, text='Seleziona cartella JSON…', command=self.on_select_folder)
        btn_select.pack(fill='x')

        ttk.Label(left, text='File rilevati:').pack(anchor='w', pady=(8, 2))
        self.listbox = tk.Listbox(left, height=20, exportselection=False)
        self.listbox.pack(fill='y', expand=False)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_page)

        self.btn_calib = ttk.Button(left, text='Calibra (2 click)', command=self.start_calibration, state='disabled')
        self.btn_calib.pack(fill='x', pady=(8, 0))

        # slider per limitare/estendere preview
        ttk.Label(left, text='Limite anteprima (strokes):').pack(anchor='w', pady=(8, 2))
        self.var_limit = tk.IntVar(value=self.PREVIEW_MAX_STROKES)
        self.spin_limit = ttk.Spinbox(left, from_=100, to=500000, increment=100, textvariable=self.var_limit, width=12, command=self._redraw_preview)
        self.spin_limit.pack(anchor='w')

        self.lbl_dims = ttk.Label(left, text='Pagina: –')
        self.lbl_dims.pack(anchor='w', pady=(8, 0))
        self.lbl_canvas = ttk.Label(left, text='Canvas TL/BR: –')
        self.lbl_canvas.pack(anchor='w')
        self.lbl_scale = ttk.Label(left, text='Scala Sx/Sy: –')
        self.lbl_scale.pack(anchor='w')

        ttk.Separator(self.phase1, orient='vertical').pack(side='left', fill='y', padx=4)

        right = ttk.Frame(self.phase1)
        right.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        # Canvas di anteprima con scroll
        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill='both', expand=True)
        self.preview = tk.Canvas(canvas_frame, bg='#f5f5f5')
        self.preview.pack(fill='both', expand=True)

        # Barra in basso con salvataggio calibrazione
        bottom = ttk.Frame(right)
        bottom.pack(fill='x')
        ttk.Button(bottom, text='Salva calibrazione…', command=self.save_calibration).pack(side='right')

        # Resize binding per ridisegnare alla dimensione nuova
        self.preview.bind('<Configure>', lambda e: self._redraw_preview())

    # -------------------------
    # Loader JSON
    # -------------------------
    def on_select_folder(self):
        start_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        path = filedialog.askdirectory(title='Seleziona cartella con JSON', initialdir=start_dir)
        if not path:
            return
        self.folder_path = path
        self.pages.clear()
        self.listbox.delete(0, tk.END)

        for fp in sorted(glob.glob(os.path.join(path, '*.json'))):
            try:
                page = self._load_page(fp)
            except Exception as e:
                print(f'Errore nel parsing {fp}: {e}')
                continue
            self.pages.append(page)
            self.listbox.insert(tk.END, os.path.basename(fp))

        if self.pages:
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self.listbox.event_generate('<<ListboxSelect>>')
            self.btn_calib.configure(state='normal')
        else:
            messagebox.showwarning('Nessun file', 'Nessun JSON trovato nella cartella selezionata.')
            self.btn_calib.configure(state='disabled')

    def _load_page(self, path: str) -> PageData:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Diversi formati possibili: image_size.width/height oppure width/height root
        image_size = data.get('image_size') or {}
        width = int(image_size.get('width') or data.get('width'))
        height = int(image_size.get('height') or data.get('height'))
        strokes_raw = data.get('strokes', [])
        strokes: List[Stroke] = []
        for s in strokes_raw:
            pts = s.get('points') or []
            if not pts:
                continue
            try:
                pts2 = [(float(x), float(y)) for x, y in pts]
            except Exception:
                continue
            closed = bool(s.get('closed', False))
            strokes.append(Stroke(points=pts2, closed=closed))
        return PageData(width=width, height=height, strokes=strokes, path=path)

    # -------------------------
    # Selezione e anteprima
    # -------------------------
    def on_select_page(self, _event=None):
        if not self.pages:
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < 0 or idx >= len(self.pages):
            return
        self.current_index = idx
        page = self.pages[idx]
        self.lbl_dims.configure(text=f'Pagina: {page.width} × {page.height} px | Strokes: {len(page.strokes)}')
        self._redraw_preview()

    def _redraw_preview(self):
        self.preview.delete('all')
        if self.current_index < 0:
            return
        page = self.pages[self.current_index]
        cw = self.preview.winfo_width()
        ch = self.preview.winfo_height()
        if cw <= 2 or ch <= 2:
            return
        # Calcola scala per fit con margine
        margin = 20
        sx = (cw - 2*margin) / max(1, page.width)
        sy = (ch - 2*margin) / max(1, page.height)
        scale = min(sx, sy)
        ox = (cw - page.width*scale) / 2
        oy = (ch - page.height*scale) / 2
        # bordo pagina
        self.preview.create_rectangle(ox, oy, ox + page.width*scale, oy + page.height*scale, outline='#c8c8c8', width=2)

        # limite configurabile
        limit = max(1, int(self.var_limit.get()))
        drawn = 0
        for s in page.strokes:
            if drawn >= limit:
                break
            pts = s.points
            n = len(pts)
            if n == 0:
                continue
            # trasformazione
            flat = []
            for (x, y) in pts:
                px = ox + x*scale
                py = oy + y*scale
                flat.extend([px, py])
            # disegno robusto: dots, segmenti corti e polilinee
            if n == 1:
                (px, py) = flat
                r = max(1.5, 1.5*scale)
                self.preview.create_oval(px-r, py-r, px+r, py+r, fill='#cc0000', outline='')
                drawn += 1
            elif n == 2:
                self.preview.create_line(*flat, fill='#cc0000', width=1, capstyle=tk.ROUND, joinstyle=tk.ROUND)
                drawn += 1
            else:
                self.preview.create_line(*flat, fill='#cc0000', width=1, capstyle=tk.ROUND, joinstyle=tk.ROUND)
                drawn += 1

    # -------------------------
    # Calibrazione
    # -------------------------
    def start_calibration(self):
        if self.current_index < 0:
            messagebox.showwarning('Seleziona una pagina', 'Seleziona prima un JSON dalla lista.')
            return
        # Mostra istruzioni prima di aprire l'overlay
        messagebox.showinfo(
            'Calibrazione',
            'Si aprirà un overlay a schermo intero.'
            'Hai 3 secondi per cambiare finestra.\n'
            'Dopo i 3 secondi, clicca:\n'
            '1) angolo Alto-Sinistra\n'
            '2) angolo Basso-Destra del foglio in Samsung Notes.\n'
            'Premi ESC per annullare.'
        )
        # Crea l'overlay DOPO la chiusura del messagebox e assicurati che sia visibile
        def _open_overlay():
            ov = CalibOverlay(self, self.on_points_captured)
            ov.wait_visibility()
            ov.lift()
            ov.focus_force()
        # Usa after invece di sleep per non bloccare la GUI
        self.after(3000, _open_overlay)

    def on_points_captured(self, tl_xy: Tuple[int, int], br_xy: Tuple[int, int]):
        # Assicura che TL e BR siano correttamente ordinati
        x1, y1 = int(tl_xy[0]), int(tl_xy[1])
        x2, y2 = int(br_xy[0]), int(br_xy[1])
        tl = (min(x1, x2), min(y1, y2))
        br = (max(x1, x2), max(y1, y2))
        self.screen_tl = tl
        self.screen_br = br
        page = self.pages[self.current_index]
        sx = (self.screen_br[0] - self.screen_tl[0]) / float(page.width)
        sy = (self.screen_br[1] - self.screen_tl[1]) / float(page.height)
        self.lbl_canvas.configure(text=f'Canvas TL: {self.screen_tl}  BR: {self.screen_br}')
        self.lbl_scale.configure(text=f'Scala Sx/Sy: {sx:.4f} / {sy:.4f}')

    def get_calibration(self) -> Optional[dict]:
        if not (self.screen_tl and self.screen_br and self.current_index >= 0):
            return None
        page = self.pages[self.current_index]
        sx = (self.screen_br[0] - self.screen_tl[0]) / float(page.width)
        sy = (self.screen_br[1] - self.screen_tl[1]) / float(page.height)
        return {
            'page_width': page.width,
            'page_height': page.height,
            'tl': [self.screen_tl[0], self.screen_tl[1]],
            'br': [self.screen_br[0], self.screen_br[1]],
            'scale': [sx, sy],
            'folder': self.folder_path,
            'json_path': page.path,
        }

    # -------------------------
    # Salvataggio calibrazione
    # -------------------------
    def save_calibration(self):
        calib = self.get_calibration()
        if not calib:
            messagebox.showwarning('Calibrazione mancante', 'Esegui la calibrazione prima di salvare.')
            return
        start_dir = self.folder_path or os.path.dirname(os.path.abspath(sys.argv[0]))
        path = filedialog.asksaveasfilename(
            title='Salva calibrazione',
            initialdir=start_dir,
            initialfile='calibration.json',
            defaultextension='.json',
            filetypes=[('JSON', '*.json')]
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(calib, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror('Errore salvataggio', str(e))
            return
        messagebox.showinfo('OK', f'Calibrazione salvata in {path}')

# -----------------------------
# Entry point
# -----------------------------

def main():
    app = App()
    app.mainloop()

if __name__ == '__main__':
    main()