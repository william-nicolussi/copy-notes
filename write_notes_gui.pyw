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
import pyautogui as pag
import threading
import time
import ctypes


from ctypes import wintypes
import threading

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



class DryRunOverlay(tk.Toplevel):
    """Overlay full‑screen semi‑trasparente per mostrare gli stroke mappati sul canvas reale
    SENZA muovere il mouse. ESC per chiudere."""
    def __init__(self, master, page: PageData, tl: Tuple[int,int], br: Tuple[int,int], limit: int = 999999, show_box: bool = True):
        super().__init__(master)
        self.page = page
        self.tl = tl
        self.br = br
        self.limit = limit
        self.show_box = show_box

        self.overrideredirect(True)
        try:
            self.attributes('-alpha', 0.35)  # velo leggero: vedi Notes sotto
        except Exception:
            pass
        self.attributes('-topmost', True)
        try:
            self.attributes('-fullscreen', True)
        except Exception:
            self.state('zoomed')

        self.canvas = tk.Canvas(self, bg='black')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Escape>', self._close)
        self.bind('<Escape>', self._close)

        self.update_idletasks()
        self.lift()
        self.focus_force()
        try:
            self.grab_set_global()
        except Exception:
            self.grab_set()

        self._draw()

    def _close(self, _evt=None):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _draw(self):
        self.canvas.delete('all')
        # bordo canvas reale
        if self.show_box:
            self.canvas.create_rectangle(self.tl[0], self.tl[1], self.br[0], self.br[1], outline='#66ff66', width=2)
        # mapping
        sx = (self.br[0] - self.tl[0]) / float(max(1, self.page.width))
        sy = (self.br[1] - self.tl[1]) / float(max(1, self.page.height))
        drawn = 0
        for s in self.page.strokes:
            if drawn >= self.limit:
                break
            pts = s.points
            n = len(pts)
            if n == 0:
                continue
            flat = []
            for (x, y) in pts:
                px = self.tl[0] + x * sx
                py = self.tl[1] + y * sy
                flat.extend([px, py])
            color = '#ff4040'
            if n == 1:
                (px, py) = flat
                r = 2
                self.canvas.create_oval(px-r, py-r, px+r, py+r, fill=color, outline='')
            else:
                self.canvas.create_line(*flat, fill=color, width=1, capstyle=tk.ROUND, joinstyle=tk.ROUND)
            drawn += 1

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

        # Fase 2 – Dry Run con calibrazione da file
        self.phase2 = ttk.Frame(self.nb)
        self.nb.add(self.phase2, text='Fase 2 – Dry Run')
        self._build_phase2()

        # Fase 3 – Disegno reale
        self.phase3 = ttk.Frame(self.nb)
        self.nb.add(self.phase3, text='Fase 3 – Disegno')
        self._build_phase3()

        
        # Stato disegno
        self._stop_flag = False
        self._draw_thread = None
# Fase 3 – Disegno reale
        self.phase3 = ttk.Frame(self.nb)
        self.nb.add(self.phase3, text='Fase 3 – Disegno')
        self._build_phase3()

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
    # Fase 2 – Dry Run con calibrazione da file
    # -------------------------
    def _build_phase2(self):
        wrapper = ttk.Frame(self.phase2)
        wrapper.pack(fill='both', expand=True, padx=8, pady=8)

        left = ttk.Frame(wrapper)
        left.pack(side='left', fill='y')

        ttk.Label(left, text='Calibrazione:').pack(anchor='w')
        self.var_calib_path = tk.StringVar(value='')
        row = ttk.Frame(left)
        row.pack(fill='x', pady=(2, 8))
        ttk.Entry(row, textvariable=self.var_calib_path, width=40).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='Scegli…', command=self.on_load_calibration_file).pack(side='left', padx=(6,0))

        self.lbl_calib_info = ttk.Label(left, text='Nessuna calibrazione caricata')
        self.lbl_calib_info.pack(anchor='w', pady=(4, 12))

        ttk.Button(left, text='Dry run (overlay 3s)…', command=self.start_dry_run).pack(fill='x')

        ttk.Label(left, text='Suggerimento: seleziona un JSON in Fase 1 per l\'anteprima/dry run.').pack(anchor='w', pady=(12, 0))

        # area vuota a destra per futura anteprima locale
        right = ttk.Frame(wrapper)
        right.pack(side='left', fill='both', expand=True, padx=12)
        ttk.Label(right, text='Questa fase non muove il mouse: mostra solo una sovrapposizione sullo schermo.').pack(anchor='w')

    def on_load_calibration_file(self):
        start_dir = self.folder_path or os.path.dirname(os.path.abspath(sys.argv[0]))
        path = filedialog.askopenfilename(
            title='Seleziona calibration.json',
            initialdir=start_dir,
            filetypes=[('JSON', '*.json')]
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror('Errore', f'Impossibile leggere il file di calibrazione:\\n{e}')
            return

        # accetta chiavi tl/br come liste [x,y] o tuple
        try:
            tl = tuple(data['tl'])
            br = tuple(data['br'])
            page_w = int(data.get('page_width', 0))
            page_h = int(data.get('page_height', 0))
        except Exception as e:
            messagebox.showerror('Errore calibrazione', f'Formato non valido: {e}')
            return

        self.screen_tl = (int(tl[0]), int(tl[1]))
        self.screen_br = (int(br[0]), int(br[1]))
        self.var_calib_path.set(path)
        self.lbl_calib_info.configure(text=f"TL={self.screen_tl}, BR={self.screen_br}, page=({page_w}×{page_h})")
        # aggiorna label anche in fase 1
        try:
            self.lbl_canvas.configure(text=f'Canvas TL/BR: {self.screen_tl} → {self.screen_br}')
            if self.current_index >= 0 and self.current_index < len(self.pages):
                page = self.pages[self.current_index]
                sx = (self.screen_br[0] - self.screen_tl[0]) / float(max(1, page.width))
                sy = (self.screen_br[1] - self.screen_tl[1]) / float(max(1, page.height))
                self.lbl_scale.configure(text=f'Scala Sx/Sy: {sx:.3f}, {sy:.3f}')
        except Exception:
            pass

    def start_dry_run(self):
        if self.current_index < 0 or self.current_index >= len(self.pages):
            messagebox.showwarning('Seleziona un JSON', 'Seleziona prima un JSON in Fase 1.')
            return
        if not (self.screen_tl and self.screen_br):
            messagebox.showwarning('Calibrazione mancante', 'Carica una calibrazione (calibration.json) in Fase 2 oppure esegui la calibrazione in Fase 1.')
            return

        # Messaggio e delay 3 secondi per cambiare finestra
        messagebox.showinfo(
            'Dry run',
            'Hai 3 secondi per cambiare finestra.\n'
            'Verrà mostrata una sovrapposizione dei tratti sullo schermo calibrato.\n'
            'Premi ESC per chiudere l\'overlay.'
        )

        def _open():
            page = self.pages[self.current_index]
            ov = DryRunOverlay(self, page, self.screen_tl, self.screen_br)
            ov.lift()
            ov.focus_force()

        self.after(3000, _open)

    # -------------------------
    # Fase 3 – Disegno reale con mouse
    # -------------------------
    def _build_phase3(self):
        left = ttk.Frame(self.phase3)
        left.pack(side='left', fill='y', padx=8, pady=8)

        self.lbl_draw_state = ttk.Label(left, text='Pronto')
        self.lbl_draw_state.pack(anchor='w', pady=(0,8))

        ttk.Label(left, text='Velocità (px/s):').pack(anchor='w')
        self.var_speed = tk.IntVar(value=1200)
        ttk.Spinbox(left, from_=100, to=10000, increment=100, textvariable=self.var_speed, width=12).pack(anchor='w')

        ttk.Label(left, text='Limite strokes:').pack(anchor='w', pady=(8,0))
        self.var_draw_limit = tk.IntVar(value=2000000)
        ttk.Spinbox(left, from_=1, to=5000000, increment=100, textvariable=self.var_draw_limit, width=12).pack(anchor='w')

        self.var_show_box3 = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text='Mostra bordo canvas', variable=self.var_show_box3).pack(anchor='w', pady=(8,0))

        ttk.Button(left, text='Start (3s delay)', command=self.start_drawing_3s).pack(fill='x', pady=(12,4))
        ttk.Button(left, text='Stop (ESC)', command=self.stop_drawing).pack(fill='x')

        right = ttk.Frame(self.phase3)
        right.pack(side='left', fill='both', expand=True, padx=12, pady=8)
        ttk.Label(right, text='Suggerimento: porta in primo piano Samsung Notes prima di premere Start. Premi ESC per fermare.').pack(anchor='nw')

        # Stato thread
        self._draw_thread = None

    def stop_drawing(self):
        self._stop_flag = True
        self.lbl_draw_state.configure(text='Stop richiesto (ESC)')

    def start_drawing_3s(self):
        if self.current_index < 0 or self.current_index >= len(self.pages):
            messagebox.showwarning('Seleziona un JSON', 'Seleziona prima un JSON in Fase 1.')
            return
        cal = self.get_calibration()
        if not cal:
            messagebox.showwarning('Calibrazione mancante', 'Esegui la calibrazione (Fase 1) o carica un calibration.json.')
            return
        messagebox.showinfo('Disegno', 'Hai 3 secondi per cambiare finestra su Samsung Notes.\nPer fermare: premi Stop oppure porta il mouse nell angolo alto-sinistra (fail-safe).')
        self.after(3000, self._start_drawing_thread)

    def _start_drawing_thread(self):
        if self._draw_thread and self._draw_thread.is_alive():
            messagebox.showwarning('In corso', 'Un disegno è già in esecuzione.')
            return
        self._stop_flag = False
        self.lbl_draw_state.configure(text='Disegno in corso…')
        self._draw_thread = threading.Thread(target=self._draw_current_page_safe, daemon=True)
        self._draw_thread.start()

    def _draw_current_page_safe(self):
        try:
            self._draw_current_page()
            self.after(0, lambda: self.lbl_draw_state.configure(text='Completato'))
        except Exception as e:
            msg = f'Errore: {e}'
            self.after(0, lambda m=msg: self.lbl_draw_state.configure(text=m))

    def _draw_current_page(self):

        page = self.pages[self.current_index]
        cal = self.get_calibration()
        tl = tuple(cal['tl']); br = tuple(cal['br'])
        sx = (br[0]-tl[0]) / float(max(1, page.width))
        sy = (br[1]-tl[1]) / float(max(1, page.height))

        speed_px_s = max(50, int(self.var_speed.get()))
        limit = int(self.var_draw_limit.get())

        # durata per segmento: distanza / velocità
        def segment_duration(p0, p1):
            x0,y0 = p0; x1,y1 = p1
            dist = ((x1-x0)**2 + (y1-y0)**2)**0.5
            return max(0.0, dist / float(speed_px_s))

        count = 0
        for s in page.strokes:
            if count >= limit: break
            pts = s.points
            if not pts: continue
            mapped = [(tl[0]+x*sx, tl[1]+y*sy) for (x,y) in pts]

            if len(mapped) == 1:
                x,y = mapped[0]
                pag.moveTo(x, y, duration=0)
                pag.click(button='left')
                count += 1
                continue

            # pen down, then move along points with durations per segment
            x0,y0 = mapped[0]
            pag.moveTo(x0, y0, duration=0)
            pag.mouseDown(button='left')
            for i in range(1, len(mapped)):
                p0 = mapped[i-1]; p1 = mapped[i]
                d = segment_duration(p0, p1)
                # pyautogui FAILSAFE consente stop rapido portando il mouse in alto-sinistra
                pag.moveTo(p1[0], p1[1], duration=d)
            pag.mouseUp(button='left')
            count += 1
        # fine

        def move_line(p0, p1):
            if MOUSE.esc_pressed() or self._stop_flag:
                return False
            x0, y0 = p0
            x1, y1 = p1
            dx = x1 - x0
            dy = y1 - y0
            dist = (dx*dx + dy*dy) ** 0.5
            if dist <= 0.5:
                MOUSE.set_pos(x1, y1)
                return True
            # passi ogni ~4 px
            step_len = 4.0
            steps = max(1, int(dist / step_len))
            total_time = dist / float(speed)
            delay = total_time / steps if steps > 0 else 0.0
            for i in range(1, steps+1):
                if MOUSE.esc_pressed() or self._stop_flag:
                    return False
                t = i / float(steps)
                xi = x0 + dx * t
                yi = y0 + dy * t
                MOUSE.set_pos(xi, yi)
                if delay > 0:
                    time.sleep(delay)
            return True

        # Itera gli strokes
        for s in page.strokes:
            if count >= limit:
                break
            pts = s.points
            n = len(pts)
            if n == 0:
                continue
            # Mapping a coordinate schermo
            mapped = [(tl[0] + x * sx, tl[1] + y * sy) for (x, y) in pts]

            if n == 1:
                # Click singolo
                MOUSE.set_pos(mapped[0][0], mapped[0][1])
                if MOUSE.esc_pressed() or self._stop_flag:
                    break
                MOUSE.click()
                count += 1
                continue

            # Muovi all'inizio e traccia
            MOUSE.set_pos(mapped[0][0], mapped[0][1])
            if MOUSE.esc_pressed() or self._stop_flag:
                break
            MOUSE.left_down()
            ok = True
            for i in range(1, n):
                ok = move_line(mapped[i-1], mapped[i])
                if not ok:
                    break
            MOUSE.left_up()
            if not ok:
                break
            count += 1

        # Fine: assicurati mouse up
        try:
            MOUSE.left_up()
        except Exception:
            pass
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
            'Clicca 1) angolo Alto-Sinistra e 2) angolo Basso-Destra del foglio in Samsung Notes.'
            'Premi ESC per annullare.'
        )
        # Crea l'overlay DOPO la chiusura del messagebox e assicurati che sia visibile
        def _open_overlay():
            ov = CalibOverlay(self, self.on_points_captured)
            ov.wait_visibility()
            ov.lift()
            ov.focus_force()
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