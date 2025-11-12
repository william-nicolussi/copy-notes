#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import threading
from core.pdf_io import run_fase1
from core.binarize import run_fase2
from core.skeletonize import run_fase3
from core.segment import run_fase4


APP_TITLE = "Strokes Exporter"
APP_SIZE = "1080x740"
MIN_SIZE = (980, 640)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(APP_SIZE)
        self.minsize(*MIN_SIZE)

        self._init_state_vars()
        self._build_menu_bar()
        self._build_body()
        self._build_statusbar()
        self._center_on_screen()

    def _init_state_vars(self):
        self.base_dir = tk.StringVar(value=str(Path(__file__).resolve().parent))
        self.pdf_path = tk.StringVar(value="")
        #self.calibration_path = tk.StringVar(value="calibration_path")

        # Parametri Fase 1
        self.p1_dpi = tk.IntVar(value=600)
        self.p1_antigrid = tk.BooleanVar(value=True)
        self.p1_s_thr = tk.StringVar(value="0.26")
        self.p1_v_thr = tk.StringVar(value="0.65")
        self.p1_thicken = tk.IntVar(value=2)

        # Parametri Fase 2
        self.p2_method = tk.StringVar(value="otsu")
        self.p2_fixed_thr = tk.StringVar(value="0.65")
        self.p2_sauvola_win = tk.IntVar(value=25)
        self.p2_sauvola_k = tk.StringVar(value="0.2")
        self.p2_remove_specks = tk.IntVar(value=0)
        self.p2_dilate_px = tk.IntVar(value=0)
        self.p2_preview = tk.BooleanVar(value=False)

        # Parametri Fase 3
        self.p3_method = tk.StringVar(value="skeletonize")
        self.p3_min_obj = tk.IntVar(value=16)
        self.p3_do_close = tk.BooleanVar(value=False)
        #self.p3_prune_px = tk.IntVar(value=0)
        self.p3_preview = tk.BooleanVar(value=True)


        # Parametri Fase 4
        #self.p4_min_len = tk.IntVar(value=0)
        #self.p4_simplify_eps = tk.StringVar(value="0.0")
        #self.p4_gap_px = tk.IntVar(value=0)
        #self.p4_resample_px = tk.IntVar(value=0)
        #self.p4_export_fmt = tk.StringVar(value="json")
        self.p4_preview = tk.BooleanVar(value=False)

        # Status
        self.status = tk.StringVar(value="Pronto")

    # ------------------ Menu ------------------
    def _build_menu_bar(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        m_help = tk.Menu(menubar, tearoff=False)
        m_help.add_command(label="Informazioni…", command=self._show_about)
        menubar.add_cascade(label="Aiuto", menu=m_help)

    # ------------------ Body ------------------
    def _build_body(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.tab1 = self._make_tab1(nb)
        self.tab2 = self._make_tab2(nb)
        self.tab3 = self._make_tab3(nb)
        self.tab4 = self._make_tab4(nb)

        nb.add(self.tab1, text="Fase 1 — PDF → cleaned")
        nb.add(self.tab2, text="Fase 2 — cleaned → BW")
        nb.add(self.tab3, text="Fase 3 — BW → Skeleton")
        nb.add(self.tab4, text="Fase 4 — Skeleton → Strokes")

    # ------------------ Statusbar ------------------
    def _build_statusbar(self):
        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.status, anchor="w").pack(side="left", padx=8, pady=4)

    # ------------------ Tab 1 ------------------
    def _make_tab1(self, parent):
        f = ttk.Frame(parent, padding=8)

        # Font generale leggermente più grande
        default_font = ("Segoe UI", 10)

        f.option_add("*TLabel.Font", default_font)
        f.option_add("*TEntry.Font", default_font)
        f.option_add("*TButton.Font", default_font)
        f.option_add("*TCheckbutton.Font", default_font)
        f.option_add("*TSpinbox.Font", default_font)

        # Layout principale
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)   # log espandibile

        # --- Input/Output
        lf = ttk.LabelFrame(f, text="Imposta input e output", labelanchor="nw", padding=6)
        lf.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(lf, text="Cartella output").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.base_dir, width=60).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(lf, text="Sfoglia…", command=self._pick_output_dir).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        ttk.Label(lf, text="File PDF").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.pdf_path, width=60).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(lf, text="Sfoglia…", command=self._pick_pdf).grid(row=1, column=2, sticky="w", padx=6, pady=4)
        lf.columnconfigure(1, weight=1)

        # --- Parametri
        params = ttk.LabelFrame(f, text="Parametri", padding=6)
        params.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(params, text="DPI").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(params, from_=72, to=1200, increment=24, textvariable=self.p1_dpi, width=8)\
            .grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(params, text="S soglia").grid(row=0, column=2, sticky="e", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.p1_s_thr, width=8)\
            .grid(row=0, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(params, text="V soglia").grid(row=0, column=4, sticky="e", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.p1_v_thr, width=8)\
            .grid(row=0, column=5, sticky="w", padx=6, pady=4)

        ttk.Label(params, text="Thicken").grid(row=0, column=6, sticky="e", padx=6, pady=4)
        ttk.Spinbox(params, from_=0, to=10, textvariable=self.p1_thicken, width=8)\
            .grid(row=0, column=7, sticky="w", padx=6, pady=4)

        # checkbox sotto
        ttk.Checkbutton(params, text="Anti-grid (rimuove quadretti)", variable=self.p1_antigrid)\
            .grid(row=1, column=0, columnspan=8, sticky="w", padx=6, pady=(2, 4))

        for c in (1, 3, 5, 7):
            params.grid_columnconfigure(c, minsize=80)
        params.grid_columnconfigure(8, weight=1)

        # --- Log (più compatta) 
        log_container = ttk.LabelFrame(f, text="Log")
        log_container.grid(row=2, column=0, sticky="nsew", pady=(6, 0))

        self.log1 = tk.Text(log_container, height=10)
        self.log1.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(log_container, orient="vertical", command=self.log1.yview)
        yscroll.pack(side="right", fill="y")
        self.log1.configure(yscrollcommand=yscroll.set)

        # --- Bottom bar per tab: status a sinistra, avvio a destra
        actions = ttk.Frame(f)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 4))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="▶ Avvia Fase 1", width=28, style="Accent.TButton",
                   command=lambda: (self._set_status("Fase 1"),self._start_fase1())).grid(row=0, column=1, sticky="e", padx=(0, 6))


        return f

    # ------------------ Tab 2 ------------------
    def _make_tab2(self, parent):
        f = ttk.Frame(parent, padding=8)

        # Font generale leggermente più grande
        default_font = ("Segoe UI", 10)

        f.option_add("*TLabel.Font", default_font)
        f.option_add("*TEntry.Font", default_font)
        f.option_add("*TButton.Font", default_font)
        f.option_add("*TCheckbutton.Font", default_font)
        f.option_add("*TSpinbox.Font", default_font)

        # Layout principale
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)   # log espandibile

        # --- Input/Output (come Tab 1)
        lf = ttk.LabelFrame(f, text="Imposta input e output", labelanchor="nw", padding=6)
        lf.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(lf, text="Cartella output").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.base_dir, width=60).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(lf, text="Sfoglia…", command=self._pick_output_dir).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        # --- Parametri personalizzati Tab 2
        params = ttk.LabelFrame(f, text="Parametri", padding=6)
        params.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # riga 1
        ttk.Label(params, text="Metodo").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Combobox(params, textvariable=self.p2_method, values=["fixed","otsu","sauvola"], state="readonly", width=12)\
        .grid(row=0, column=1, sticky="w", padx=6, pady=4)


        ttk.Label(params, text="Soglia fissa").grid(row=0, column=2, sticky="e", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.p2_fixed_thr, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=4)


        ttk.Label(params, text="Sauvola win").grid(row=0, column=4, sticky="e", padx=6, pady=4)
        ttk.Spinbox(params, from_=1, to=199, increment=2, textvariable=self.p2_sauvola_win, width=6).grid(row=0, column=5, sticky="w", padx=6, pady=4)


        # riga 2
        ttk.Label(params, text="Sauvola k").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.p2_sauvola_k, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=4)


        ttk.Label(params, text="Remove specks").grid(row=1, column=2, sticky="e", padx=6, pady=4)
        ttk.Spinbox(params, from_=0, to=9999, textvariable=self.p2_remove_specks, width=8).grid(row=1, column=3, sticky="w", padx=6, pady=4)


        ttk.Label(params, text="Dilate px").grid(row=1, column=4, sticky="e", padx=6, pady=4)
        ttk.Spinbox(params, from_=0, to=64, textvariable=self.p2_dilate_px, width=6).grid(row=1, column=5, sticky="w", padx=6, pady=4)


        # riga 3
        ttk.Checkbutton(params, text="See preview (bw_preview)", variable=self.p2_preview).grid(row=2, column=0, columnspan=6, sticky="w", padx=6, pady=(2,4))


        for c in range(6):
            params.grid_columnconfigure(c, weight=0)
        params.grid_columnconfigure(6, weight=1)

        # --- Log compatta
        log_container = ttk.LabelFrame(f, text="Log")
        log_container.grid(row=2, column=0, sticky="nsew", pady=(6, 0))

        self.log2 = tk.Text(log_container, height=10)
        self.log2.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(log_container, orient="vertical", command=self.log2.yview)
        yscroll.pack(side="right", fill="y")
        self.log1.configure(yscrollcommand=yscroll.set)


        # --- Bottom bar: status a sinistra, bottone a destra
        actions = ttk.Frame(f)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 4))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="▶ Avvia Fase 2", width=28, style="Accent.TButton",
                    command=lambda: (self._set_status("Fase 2"),self._start_fase2())).grid(row=0, column=1, sticky="e", padx=(0, 6))

        return f

    # ------------------ Tab 3 ------------------
    def _make_tab3(self, parent):
        f = ttk.Frame(parent, padding=8)

        # Font generale leggermente più grande
        default_font = ("Segoe UI", 10)

        f.option_add("*TLabel.Font", default_font)
        f.option_add("*TEntry.Font", default_font)
        f.option_add("*TButton.Font", default_font)
        f.option_add("*TCheckbutton.Font", default_font)
        f.option_add("*TSpinbox.Font", default_font)

        # Layout principale
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)   # log espandibile

        # --- Input/Output (come Tab 1)
        lf = ttk.LabelFrame(f, text="Imposta input e output", labelanchor="nw", padding=6)
        lf.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(lf, text="Cartella output").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.base_dir, width=60).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(lf, text="Sfoglia…", command=self._pick_output_dir).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        # --- Parametri personalizzati Tab 3
        params = ttk.LabelFrame(f, text="Parametri", padding=6)
        params.grid(row=1, column=0, sticky="ew", pady=(8, 0))


        ttk.Label(params, text="Metodo").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Combobox(params, textvariable=self.p3_method, values=["skeletonize","medial_axis"], state="readonly", width=14)\
        .grid(row=0, column=1, sticky="w", padx=6, pady=4)


        ttk.Label(params, text="Min obj").grid(row=0, column=2, sticky="e", padx=6, pady=4)
        ttk.Spinbox(params, from_=0, to=9999, textvariable=self.p3_min_obj, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=4)


        #ttk.Label(params, text="Prune px").grid(row=0, column=4, sticky="e", padx=6, pady=4)
        #ttk.Spinbox(params, from_=0, to=100, textvariable=self.p3_prune_px, width=6).grid(row=0, column=5, sticky="w", padx=6, pady=4)


        ttk.Checkbutton(params, text="Closing 3×3", variable=self.p3_do_close).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(params, text="See preview (skel_preview)", variable=self.p3_preview).grid(row=1, column=2, columnspan=4, sticky="w", padx=6, pady=4)


        for c in range(6):
            params.grid_columnconfigure(c, weight=0)
        params.grid_columnconfigure(6, weight=1)

        # --- Log compatta
        log_container = ttk.LabelFrame(f, text="Log")
        log_container.grid(row=2, column=0, sticky="nsew", pady=(6, 0))

        self.log3 = tk.Text(log_container, height=10)
        self.log3.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(log_container, orient="vertical", command=self.log3.yview)
        yscroll.pack(side="right", fill="y")
        self.log3.configure(yscrollcommand=yscroll.set)



        # --- Bottom bar: status a sinistra, bottone a destra
        actions = ttk.Frame(f)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 4))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="▶ Avvia Fase 3", width=28, style="Accent.TButton",
                   command=lambda: (self._set_status("Fase 3"),self._start_fase3())).grid(row=0, column=1, sticky="e", padx=(0, 6))

        return f

    # ------------------ Tab 4 ------------------
    def _make_tab4(self, parent):
        f = ttk.Frame(parent, padding=8)

        # Font generale leggermente più grande
        default_font = ("Segoe UI", 10)

        f.option_add("*TLabel.Font", default_font)
        f.option_add("*TEntry.Font", default_font)
        f.option_add("*TButton.Font", default_font)
        f.option_add("*TCheckbutton.Font", default_font)
        f.option_add("*TSpinbox.Font", default_font)

        # Layout principale
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)   # log espandibile

        # --- Input/Output (come Tab 1)
        lf = ttk.LabelFrame(f, text="Imposta input e output", labelanchor="nw", padding=6)
        lf.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(lf, text="Cartella output").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.base_dir, width=60).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(lf, text="Sfoglia…", command=self._pick_output_dir).grid(row=0, column=2, sticky="w", padx=6, pady=4)


        # --- Parametri personalizzati Tab 4
        params = ttk.LabelFrame(f, text="Parametri", padding=6)
        params.grid(row=1, column=0, sticky="ew", pady=(8, 0))


        #ttk.Label(params, text="Min stroke len").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        #ttk.Spinbox(params, from_=0, to=9999, textvariable=self.p4_min_len, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=4)


        #ttk.Label(params, text="Simplify ε (px)").grid(row=0, column=2, sticky="e", padx=6, pady=4)
        #ttk.Entry(params, textvariable=self.p4_simplify_eps, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=4)


        #ttk.Label(params, text="Connect gaps ≤ px").grid(row=0, column=4, sticky="e", padx=6, pady=4)
        #ttk.Spinbox(params, from_=0, to=64, textvariable=self.p4_gap_px, width=6).grid(row=0, column=5, sticky="w", padx=6, pady=4)


        #ttk.Label(params, text="Resample step px").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        #ttk.Spinbox(params, from_=0, to=64, textvariable=self.p4_resample_px, width=6).grid(row=1, column=1, sticky="w", padx=6, pady=4)


        #ttk.Label(params, text="Export format").grid(row=1, column=2, sticky="e", padx=6, pady=4)
        #ttk.Combobox(params, textvariable=self.p4_export_fmt, values=["json","ndjson"], state="readonly", width=8).grid(row=1, column=3, sticky="w", padx=6, pady=4)


        ttk.Checkbutton(params, text="See preview (strokes_preview)", variable=self.p4_preview).grid(row=2, column=0, columnspan=6, sticky="w", padx=6, pady=4)


        for c in range(6):
            params.grid_columnconfigure(c, weight=0)
        params.grid_columnconfigure(6, weight=1)

        # --- Log compatta
        log_container = ttk.LabelFrame(f, text="Log")
        log_container.grid(row=2, column=0, sticky="nsew", pady=(6, 0))

        self.log4 = tk.Text(log_container, height=10)
        self.log4.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(log_container, orient="vertical", command=self.log4.yview)
        yscroll.pack(side="right", fill="y")
        self.log4.configure(yscrollcommand=yscroll.set)


        # --- Bottom bar: status a sinistra, bottone a destra
        actions = ttk.Frame(f)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 4))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="▶ Avvia Fase 4", width=28, style="Accent.TButton",
                   command=lambda: (self._set_status("Fase 4"),self._start_fase4())).grid(row=0, column=1, sticky="e", padx=(0, 6))

        return f

    # ------------------ Helpers ------------------
    def _show_about(self):
        messagebox.showinfo(APP_TITLE, "Strokes Exporter\nDemo GUI")

    def _set_status(self, msg: str):
        self.status.set(msg)

    def _pick_output_dir(self):
        d = filedialog.askdirectory(initialdir=self.base_dir.get() or ".")
        if d:
            self.base_dir.set(d)
            self._set_status(f"Cartella output: {d}")
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showwarning("Attenzione", f"Impossibile creare cartella:\n{e}")

    def _pick_pdf(self):
        f = filedialog.askopenfilename(
            title="Seleziona PDF",
            filetypes=[["PDF", "*.pdf"], ["Tutti i file", "*.*"]],
            initialdir=str(Path(self.base_dir.get() or ".").resolve())
        )
        if f:
            self.pdf_path.set(f)
            self._set_status(f"Selezionato PDF: {f}")

    def _open_output_dir(self):
        p = Path(self.base_dir.get() or ".").resolve()
        try:
            if sys.platform.startswith("win"):
                import os
                os.startfile(p)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", str(p)], check=False)
            else:
                import subprocess
                subprocess.run(["xdg-open", str(p)], check=False)
        except Exception as e:
            messagebox.showwarning("Attenzione", f"Impossibile aprire la cartella:\n{e}")

    def _log(self, widget, msg):
        if not widget: return
        widget.insert("end", msg + "\n")
        widget.see("end")

    def _params_phase1(self):
        return {
            "dpi": int(self.p1_dpi.get()),
            "anti_grid": bool(self.p1_antigrid.get()),
            "s_thr": float(self.p1_s_thr.get()),
            "v_thr": float(self.p1_v_thr.get()),
            "thicken": int(self.p1_thicken.get()),
        }

    def _start_fase1(self):
        base_dir = Path(self.base_dir.get()).resolve()
        pdf_path = Path(self.pdf_path.get()) if self.pdf_path.get() else None
        params = self._params_phase1()
        self._set_status("Fase 1 in esecuzione…")
        self._log(self.log1, f"[F1] Avvio con parametri: {params}")

        def job():
            try:
                run_fase1(
                    base_dir=base_dir,
                    pdf_path=pdf_path,
                    params=params,
                    logger=lambda m: self.log1.after(0, self._log, self.log1, m),
                )
            except Exception as e:
                self.log1.after(0, self._log, self.log1, f"[ERR] {e}")
            finally:
                self.status.set("Pronto")

        threading.Thread(target=job, daemon=True).start()

    def _params_phase2(self):
        return {
            "method": self.p2_method.get(),
            "fixed_thr": float(self.p2_fixed_thr.get()),
            "sau_w": int(self.p2_sauvola_win.get()),
            "sau_k": float(self.p2_sauvola_k.get()),
            "specks": int(self.p2_remove_specks.get()),
            "dilate": int(self.p2_dilate_px.get()),
            "preview": bool(self.p2_preview.get()),
        }

    def _start_fase2(self):
        base_dir = Path(self.base_dir.get()).resolve()
        params = self._params_phase2()
        self._set_status("Fase 2 in esecuzione…")
        self._log(self.log2, f"[F2] Avvio con parametri: {params}")

        def job():
            try:
                run_fase2(
                    base_dir=base_dir,
                    params=params,
                    logger=lambda m: self.log2.after(0, self._log, self.log2, m),
                )
            except Exception as e:
                self.log2.after(0, self._log, self.log2, f"[ERR] {e}")
            finally:
                self.status.set("Pronto")

        threading.Thread(target=job, daemon=True).start()

    def _params_phase3(self):
        return {
            "method": self.p3_method.get(),
            "min_obj": int(self.p3_min_obj.get()),
            "do_close": bool(self.p3_do_close.get()),
            "preview": bool(self.p3_preview.get()),
        }
    
    def _start_fase3(self):
        base_dir = Path(self.base_dir.get()).resolve()
        params = self._params_phase3()
        self._set_status("Fase 3 in esecuzione…")
        self._log(self.log3, f"[F3] Avvio con parametri: {params}")

        def job():
            try:
                run_fase3(
                    base_dir=base_dir,
                    params=params,
                    logger=lambda m: self.log3.after(0, self._log, self.log3, m),
                )
            except Exception as e:
                self.log3.after(0, self._log, self.log3, f"[ERR] {e}")
            finally:
                self.status.set("Pronto")

        threading.Thread(target=job, daemon=True).start()


    def _params_phase4(self):
        return {
            "preview": bool(self.p4_preview.get()),
        }

    def _start_fase4(self):
        base_dir = Path(self.base_dir.get()).resolve()
        params = self._params_phase4()
        self._set_status("Fase 4 in esecuzione…")
        self._log(self.log4, f"[F4] Avvio con parametri: {params}")

        def job():
            try:
                run_fase4(
                    base_dir=base_dir,
                    params=params,
                    logger=lambda m: self.log4.after(0, self._log, self.log4, m),
                )
            except Exception as e:
                self.log4.after(0, self._log, self.log4, f"[ERR] {e}")
            finally:
                self.status.set("Pronto")

        threading.Thread(target=job, daemon=True).start()



    def _center_on_screen(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = max((sw - w) // 2, 0)
        y = max((sh - h) // 2, 0)
        self.geometry(f"{w}x{h}+{x}+{y}")

if __name__ == "__main__":
    App().mainloop()
