from dataclasses import dataclass, field
import tkinter as tk

@dataclass
class Settings:
    dpi: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=600))
    anti_grid: tk.BooleanVar = field(default_factory=lambda: tk.BooleanVar(value=True))
    s_thr: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.26))
    v_thr: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.65))
    thicken: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=2))

    bw_mode: tk.StringVar = field(default_factory=lambda: tk.StringVar(value="otsu"))
    fixed_thr: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.65))
    sau_w: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=25))
    sau_k: tk.DoubleVar = field(default_factory=lambda: tk.DoubleVar(value=0.2))
    specks: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=0))
    dilate: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=0))
    preview_bw: tk.BooleanVar = field(default_factory=lambda: tk.BooleanVar(value=False))

    sk_method: tk.StringVar = field(default_factory=lambda: tk.StringVar(value="skeletonize"))
    min_obj: tk.IntVar = field(default_factory=lambda: tk.IntVar(value=16))
    do_close: tk.BooleanVar = field(default_factory=lambda: tk.BooleanVar(value=False))

    preview_strokes: tk.BooleanVar = field(default_factory=lambda: tk.BooleanVar(value=True))
