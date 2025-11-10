#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Clean Notes — Fase 3 (Skeleton via scikit-image)

Implements skeletonization as described in scikit-image's example:
- skimage.morphology.skeletonize (Zhang-Suen-like topological thinning)
- skimage.morphology.medial_axis(return_distance=True) to get the skeleton
  AND the distance map (local half-thickness), useful to prune thin spurs.

Docs:
- Skeletonize example: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html
- skimage.morphology API: https://scikit-image.org/docs/stable/api/skimage.morphology.html

Requires: pip install scikit-image opencv-python numpy
"""
import time
import math
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis, binary_closing, remove_small_objects, footprint_rectangle
from skimage.util import invert
from skimage.filters import threshold_otsu

def to_binary_bool(img_bgr):
    """Return boolean image (True=foreground ink) from BGR/gray/BW."""
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    # Ensure background white; if border is dark, invert
    border = np.concatenate([gray[0,:], gray[-1,:], gray[:,0], gray[:,-1]])
    if border.mean() < 127:
        gray = 255 - gray
    if not np.isin(gray, [0,255]).all():
        thr = threshold_otsu(gray)
        bw = gray <= thr  # True=ink (dark)
    else:
        bw = gray == 0    # True=ink
    return bw.astype(bool)

def bool_to_png(bool_img):
    return np.where(bool_img, 0, 255).astype(np.uint8)  # True -> black ink

def prune_by_length(skel_bool, max_len=8):
    if max_len <= 0:
        return skel_bool
    sk = skel_bool.copy()
    h, w = sk.shape
    # degree map with 8-connectivity
    k = np.ones((3,3), np.uint8)
    ink_u8 = sk.astype(np.uint8)
    deg = cv2.filter2D(ink_u8, -1, k, borderType=cv2.BORDER_CONSTANT) - ink_u8
    endpoints = np.argwhere((sk) & (deg==1))
    removed_any = True
    while removed_any:
        removed_any = False
        deg = cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - sk.astype(np.uint8)
        endpoints = np.argwhere((sk) & (deg==1))
        for y,x in endpoints:
            cy,cx=y,x
            path=[(y,x)]
            steps=0
            while steps<max_len:
                nbrs=[]
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dy==0 and dx==0: 
                            continue
                        ny,nx=cy+dy,cx+dx
                        if ny<0 or nx<0 or ny>=h or nx>=w: 
                            continue
                        if sk[ny,nx]:
                            nbrs.append((ny,nx))
                if len(path)>=2 and path[-2] in nbrs:
                    nbrs.remove(path[-2])
                if len(nbrs)!=1:
                    break
                cy,cx=nbrs[0]
                path.append((cy,cx))
                steps+=1
            # erase path
            for py,px in path:
                sk[py,px]=False
            removed_any=True
    return sk

def bridge_endpoints(skel_bool, max_dist=2):
    if max_dist<=0: 
        return skel_bool
    sk = skel_bool.copy()
    k = np.ones((3,3), np.uint8)
    deg = cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - sk.astype(np.uint8)
    endpoints = np.argwhere((sk) & (deg==1))
    for i in range(len(endpoints)):
        y1,x1=endpoints[i]
        for j in range(i+1,len(endpoints)):
            y2,x2=endpoints[j]
            d = math.hypot(x2-x1,y2-y1)
            if d<=max_dist:
                # draw Bresenham
                x,y=x1,y1
                dx=abs(x2-x1); sx=1 if x1<x2 else -1
                dy=-abs(y2-y1); sy=1 if y1<y2 else -1
                err=dx+dy
                while True:
                    sk[y,x]=True
                    if x==x2 and y==y2: break
                    e2=2*err
                    if e2>=dy: err+=dy; x+=sx
                    if e2<=dx: err+=dx; y+=sy
    return sk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clean Notes — Fase 3 (scikit-image skeleton)")
        self.geometry("820x560")
        self.base_dir = tk.StringVar(value="")
        self.method = tk.StringVar(value="skeletonize")  # or medial_axis
        self.min_obj = tk.IntVar(value=0)
        self.closing = tk.BooleanVar(value=True)
        self.spur = tk.IntVar(value=0)
        self.gap = tk.IntVar(value=2)
        self.width_thr = tk.DoubleVar(value=0.0)  # only for medial_axis: prune where distance<thr

        frm=ttk.Frame(self, padding=10); frm.pack(fill="both", expand=True)
        row=0
        ttk.Label(frm, text="Cartella base (contiene 'bw')").grid(row=row, column=0, sticky='w')
        row+=1
        pfrm=ttk.Frame(frm); pfrm.grid(row=row, column=0, sticky='we', pady=4); pfrm.columnconfigure(0, weight=1)
        ttk.Entry(pfrm, textvariable=self.base_dir).grid(row=0, column=0, sticky='we')
        ttk.Button(pfrm, text="Sfoglia…", command=self.pick).grid(row=0, column=1, padx=6)
        row+=1

        opt=ttk.LabelFrame(frm, text="Skeleton options (scikit-image)"); opt.grid(row=row, column=0, sticky='we', pady=8)
        for c in range(6): opt.columnconfigure(c,weight=1)
        ttk.Label(opt, text="Metodo").grid(row=0, column=0, sticky='w')
        ttk.Combobox(opt, textvariable=self.method, values=["skeletonize","medial_axis"], width=14, state="readonly").grid(row=0, column=1, sticky='w')
        ttk.Label(opt, text="Remove small obj ≥").grid(row=0, column=2, sticky='e')
        ttk.Entry(opt, textvariable=self.min_obj, width=6).grid(row=0, column=3, sticky='w')
        ttk.Checkbutton(opt, text="Binary closing", variable=self.closing).grid(row=0, column=4, sticky='w')

        ttk.Label(opt, text="Prune spur ≤").grid(row=1, column=0, sticky='e')
        ttk.Entry(opt, textvariable=self.spur, width=6).grid(row=1, column=1, sticky='w')
        ttk.Label(opt, text="Bridge dist ≤").grid(row=1, column=2, sticky='e')
        ttk.Entry(opt, textvariable=self.gap, width=6).grid(row=1, column=3, sticky='w')
        ttk.Label(opt, text="(medial) width thr ≥").grid(row=1, column=4, sticky='e')
        ttk.Entry(opt, textvariable=self.width_thr, width=6).grid(row=1, column=5, sticky='w')

        row+=1
        ttk.Button(frm, text="Esegui: BW -> Skeleton (bw_skel)", command=self.run).grid(row=row, column=0, sticky='we', pady=10)
        row+=1
        self.pbar=ttk.Progressbar(frm, mode='determinate'); self.pbar.grid(row=row, column=0, sticky='we')
        row+=1
        self.log=tk.Text(frm, height=16, wrap='word'); self.log.grid(row=row, column=0, sticky='nsew')
        frm.rowconfigure(row, weight=1)

    def pick(self):
        d=filedialog.askdirectory(title="Scegli cartella base")
        if d: self.base_dir.set(d)

    def _log(self, s):
        self.log.insert('end', s+"\n"); self.log.see('end'); self.update_idletasks()

    def run(self):
        base = Path(self.base_dir.get().strip() or '.').resolve()
        d_bw = base / 'bw'
        if not d_bw.exists():
            messagebox.showerror("Errore", f"Manca la cartella 'bw':\n{d_bw}"); return
        out = base / 'bw_skel'; out.mkdir(exist_ok=True, parents=True)
        imgs = sorted([p for p in d_bw.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')])
        if not imgs:
            messagebox.showerror("Errore", "Nessuna immagine trovata in 'bw'."); return
        self.pbar.config(maximum=len(imgs), value=0)
        self.log.delete('1.0','end')

        method = self.method.get()
        min_obj = int(self.min_obj.get())
        do_close = bool(self.closing.get())
        spur = int(self.spur.get())
        gap = int(self.gap.get())
        width_thr = float(self.width_thr.get())

        def worker():
            t0=time.perf_counter()
            for i,p in enumerate(imgs, start=1):
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                bw_bool = to_binary_bool(bgr)
                # Optional cleanups before skeletonization
                if do_close:
                    from skimage.morphology import binary_closing, footprint_rectangle
                    bw_bool = binary_closing(bw_bool, footprint=footprint_rectangle((3,3)))
                if min_obj>0:
                    bw_bool = remove_small_objects(bw_bool, min_size=min_obj)

                if method == 'skeletonize':
                    skel = skeletonize(bw_bool)
                else:
                    skel, dist = medial_axis(bw_bool, return_distance=True)
                    if width_thr>0:
                        # keep only skeleton points with radius >= width_thr
                        skel = skel & (dist >= width_thr)

                # Post-processing similar to our previous GUI
                if spur>0:
                    skel = prune_by_length(skel, max_len=spur)
                if gap>0:
                    skel = bridge_endpoints(skel, max_dist=gap)

                outp = out / f"{p.stem}_skel.png"
                cv2.imwrite(str(outp), bool_to_png(skel))
                self._log(f"[OK] {outp.name}")
                self.pbar['value']=i

            self._log("\n[FINITO] Skeleton salvati in 'bw_skel'.")
            messagebox.showinfo("Completato", "Skeleton creati con scikit-image.")

        threading.Thread(target=worker, daemon=True).start()

if __name__ == '__main__':
    App().mainloop()
