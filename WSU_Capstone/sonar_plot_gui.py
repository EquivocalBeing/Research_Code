import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import glob
import re
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.io as pio
from scipy.ndimage import uniform_filter1d

pio.renderers.default = 'browser'

# ── Constants ─────────────────────────────────────────────────────────────────
SHOT_SIZE  = 384
ECHO_BYTES = 252

# ── Sonar helpers ──────────────────────────────────────────────────────────────
def decode_head_position(b5, b6):
    hi = (b6 & 0x3E) >> 1
    lo = ((b6 & 0x01) << 7) | (b5 & 0x7F)
    return (hi << 8) | lo

def decode_profile_range(b8, b9):
    hi = (b9 & 0x7E) >> 1
    lo = ((b9 & 0x01) << 7) | (b8 & 0x7F)
    return (hi << 8) | lo

def natural_sort_key(path):
    name  = os.path.basename(path)
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

# ── File conversion ────────────────────────────────────────────────────────────
def convert_81A_to_csv(input_file, output_csv):
    rows = []
    with open(input_file, "rb") as fh:
        while True:
            shot = fh.read(SHOT_SIZE)
            if len(shot) < SHOT_SIZE:
                break
            shot      = list(shot)
            id_str    = bytes(shot[0:3]).decode("ascii", errors="ignore")
            shotBytes = (shot[4] << 8) | shot[5]
            nToRead   = (shot[6] << 8) | shot[7]
            date      = bytes(shot[8:20]).decode("ascii",  errors="ignore")
            time      = bytes(shot[20:29]).decode("ascii", errors="ignore")
            hund      = bytes(shot[29:33]).decode("ascii", errors="ignore")
            rh        = shot[100:116]
            range_m   = rh[7]
            headPos   = decode_head_position(rh[5], rh[6])
            angle_deg = 0.3 * (headPos - 600)
            profUnits = decode_profile_range(rh[8], rh[9])
            prof_m    = profUnits * 0.002
            row = {
                "ID": id_str, "ShotBytes": shotBytes, "N_To_Read": nToRead,
                "Date": date, "Time": time, "Hundredths": hund,
                "Range_m": range_m, "HeadPos": headPos,
                "Angle_deg": round(angle_deg, 2),
                "ProfileUnits": profUnits, "Profile_m": round(prof_m, 4),
            }
            for i in range(ECHO_BYTES):
                row[f"Echo_{i}"] = shot[112 + i]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df

# ── Echo filter ────────────────────────────────────────────────────────────────
def apply_echo_filter(echo_array, filter_type=None, filter_size=5):
    if filter_type is None:
        return echo_array.astype(float)
    smoothed = uniform_filter1d(echo_array.astype(float), size=filter_size)
    if filter_type == 'lowpass':
        return smoothed
    elif filter_type == 'highpass':
        return echo_array.astype(float) - smoothed
    elif filter_type == 'bandpass':
        lp = uniform_filter1d(echo_array.astype(float), size=filter_size)
        return lp - uniform_filter1d(lp, size=filter_size * 3)
    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'.")

# ── Point cloud ────────────────────────────────────────────────────────────────
def build_point_cloud(csv_files_with_azimuths,
                      intensity_threshold=5,
                      filter_type='bandpass',
                      filter_size=1,
                      xlim=None, ylim=None, zlim=None,
                      progress_cb=None):
    all_points = []
    total = len(csv_files_with_azimuths)
    for idx, (csv_path, azimuth_deg) in enumerate(csv_files_with_azimuths):
        if progress_cb:
            progress_cb(idx, total, os.path.basename(csv_path))
        data         = pd.read_csv(csv_path)
        echo_cols    = [f"Echo_{i}" for i in range(ECHO_BYTES)]
        echo_matrix  = data[echo_cols].values
        sonar_angles = data["Angle_deg"].values
        range_metres = data["Range_m"].values
        azimuth_rad  = np.deg2rad(azimuth_deg)
        for shot_idx, theta_deg in enumerate(sonar_angles):
            theta_rad   = np.deg2rad(theta_deg)
            intensities = apply_echo_filter(echo_matrix[shot_idx], filter_type, filter_size)
            shot_range  = range_metres[shot_idx]
            r_values    = np.linspace(0, shot_range, ECHO_BYTES)
            for r, intensity in zip(r_values, intensities):
                if intensity < intensity_threshold:
                    continue
                horiz = r * np.sin(theta_rad)
                x =  horiz * np.cos(azimuth_rad)
                y = -horiz * np.sin(azimuth_rad)
                z =  r     * np.cos(theta_rad)
                if xlim and not (xlim[0] <= x <= xlim[1]): continue
                if ylim and not (ylim[0] <= y <= ylim[1]): continue
                if zlim and not (zlim[0] <= z <= zlim[1]): continue
                all_points.append([x, y, z, intensity])
    if progress_cb:
        progress_cb(total, total, "Done")
    return np.array(all_points) if all_points else np.empty((0, 4))

# ── 2-D polar plot ─────────────────────────────────────────────────────────────
def plot_2d(csv_path, colorscale='viridis', save_path=None):
    data      = pd.read_csv(csv_path)
    angles    = np.deg2rad(data['Angle_deg'].values)
    echo_cols = [f'Echo_{i}' for i in range(ECHO_BYTES)]
    points    = data[echo_cols].values
    range_m   = data['Range_m'].iloc[0]
    date      = data['Date'].iloc[0]
    r         = np.linspace(0, range_m, points.shape[1])
    Theta, R  = np.meshgrid(angles, r, indexing='ij')
    warnings.simplefilter('ignore')
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    c  = ax.pcolormesh(Theta, R, points, shading='auto', cmap=colorscale)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Intensity [NA]")
    plt.title(f"Scan Date: {date}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ── 3-D interactive plot ───────────────────────────────────────────────────────
def plot_3d(points, color_by='z', colorscale='Viridis',
            title="360° Sonar Point Cloud"):
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    color_values = z if color_by.lower() == 'z' else intensity
    color_label  = 'Z Depth [m]' if color_by.lower() == 'z' else 'Intensity [NA]'
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=1.5, color=color_values,
                    colorscale=colorscale,
                    colorbar=dict(title=color_label),
                    opacity=1),
    )])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]',
                   zaxis_title='Z [m]', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()

# ── PLY export ─────────────────────────────────────────────────────────────────
def export_ply(points, filepath):
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\n"
        "property float z\nproperty float intensity\n"
        "end_header\n"
    )
    with open(filepath, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f")
    return filepath

# ── Diagnostics ────────────────────────────────────────────────────────────────
def diagnostics_text(points):
    if len(points) == 0:
        return "WARNING: No points passed the filters!\nTry loosening threshold / limits."
    lines = [
        f"Total points : {len(points):,}",
        f"X range      : {points[:,0].min():.2f} → {points[:,0].max():.2f} m",
        f"Y range      : {points[:,1].min():.2f} → {points[:,1].max():.2f} m",
        f"Z range      : {points[:,2].min():.2f} → {points[:,2].max():.2f} m",
        f"Intensity    : {points[:,3].min():.1f} → {points[:,3].max():.1f}",
        "", "Intensity distribution:",
    ]
    bins = [0, 10, 20, 30, 50, 75, 100, 128]
    hist, edges = np.histogram(points[:,3], bins=bins)
    for i in range(len(hist)):
        lines.append(f"  {int(edges[i]):>3} - {int(edges[i+1]):>3}: {hist[i]:>10,}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════
class SonarApp(tk.Tk):
    BG       = "#0d1117"
    PANEL    = "#161b22"
    BORDER   = "#30363d"
    ACCENT   = "#58a6ff"
    ACCENT2  = "#3fb950"
    TEXT     = "#e6edf3"
    SUBTEXT  = "#8b949e"
    ENTRY_BG = "#21262d"
    BTN_BG   = "#21262d"
    BTN_HOV  = "#30363d"

    def __init__(self):
        super().__init__()
        self.title("Sonar Point Cloud Studio")
        self.configure(bg=self.BG)
        self.resizable(True, True)
        self.minsize(920, 700)

        self._csv_folder     = tk.StringVar()
        self._81a_folder     = tk.StringVar()
        self._converted_csvs = []
        self._listbox_paths  = []   # parallel list to listbox entries
        self._last_points    = None

        self._build_styles()
        self._build_ui()

    # ── styles ─────────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure(".", background=self.BG, foreground=self.TEXT,
                    font=("Courier New", 10))
        s.configure("TFrame",        background=self.BG)
        s.configure("Panel.TFrame",  background=self.PANEL, relief="flat")
        s.configure("TLabel",        background=self.BG,    foreground=self.TEXT)
        s.configure("Sub.TLabel",    background=self.PANEL, foreground=self.SUBTEXT,
                     font=("Courier New", 9))
        s.configure("Head.TLabel",   background=self.PANEL, foreground=self.ACCENT,
                     font=("Courier New", 11, "bold"))
        s.configure("TEntry",        fieldbackground=self.ENTRY_BG, foreground=self.TEXT,
                     insertcolor=self.TEXT, bordercolor=self.BORDER,
                     lightcolor=self.BORDER, darkcolor=self.BORDER)
        s.configure("TCombobox",     fieldbackground=self.ENTRY_BG, foreground=self.TEXT,
                     background=self.ENTRY_BG, selectbackground=self.ENTRY_BG,
                     arrowcolor=self.ACCENT)
        s.map("TCombobox", fieldbackground=[("readonly", self.ENTRY_BG)])
        s.configure("TCheckbutton",  background=self.PANEL, foreground=self.TEXT)
        s.configure("TNotebook",     background=self.BG, bordercolor=self.BORDER,
                     tabmargins=[2,2,0,0])
        s.configure("TNotebook.Tab", background=self.BTN_BG, foreground=self.SUBTEXT,
                     padding=[14,6], font=("Courier New", 10))
        s.map("TNotebook.Tab",
              background=[("selected", self.PANEL)],
              foreground=[("selected", self.ACCENT)])
        s.configure("Accent.TButton", background=self.ACCENT, foreground="#0d1117",
                     font=("Courier New", 10, "bold"), borderwidth=0)
        s.map("Accent.TButton",
              background=[("active","#79c0ff"),("pressed","#388bfd")])
        s.configure("Green.TButton",  background=self.ACCENT2, foreground="#0d1117",
                     font=("Courier New", 10, "bold"), borderwidth=0)
        s.map("Green.TButton",
              background=[("active","#56d364"),("pressed","#2ea043")])
        s.configure("Muted.TButton",  background=self.BTN_BG, foreground=self.TEXT,
                     borderwidth=1, relief="flat")
        s.map("Muted.TButton", background=[("active", self.BTN_HOV)])
        s.configure("Horizontal.TProgressbar",
                     troughcolor=self.ENTRY_BG, background=self.ACCENT,
                     borderwidth=0, lightcolor=self.ACCENT, darkcolor=self.ACCENT)

    # ── widget helpers ─────────────────────────────────────────────────────────
    def _field(self, parent, label, default, width=12):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill='x', pady=2)
        ttk.Label(row, text=label, style="Sub.TLabel", width=26).pack(side='left')
        var = tk.StringVar(value=str(default))
        ttk.Entry(row, textvariable=var, width=width).pack(side='left', padx=(4,0))
        return var

    def _combo(self, parent, label, choices, default, width=14):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill='x', pady=2)
        ttk.Label(row, text=label, style="Sub.TLabel", width=26).pack(side='left')
        var = tk.StringVar(value=default)
        ttk.Combobox(row, textvariable=var, values=choices,
                     width=width, state='readonly').pack(side='left', padx=(4,0))
        return var

    def _check(self, parent, label, default=False):
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(parent, text=label, variable=var,
                        style="TCheckbutton").pack(anchor='w', pady=2)
        return var

    def _section(self, parent, title):
        f = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        f.pack(fill='x', padx=6, pady=6)
        ttk.Label(f, text=title, style="Head.TLabel").pack(anchor='w', pady=(0,6))
        return f

    def _folder_row(self, parent, label, var, cb):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill='x', pady=3)
        ttk.Label(row, text=label, style="Sub.TLabel", width=20).pack(side='left')
        ttk.Entry(row, textvariable=var, width=36).pack(side='left', padx=4)
        ttk.Button(row, text="Browse…", style="Muted.TButton", command=cb).pack(side='left')

    # ── top-level layout ───────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self, bg=self.BG, pady=10)
        hdr.pack(fill='x', padx=20, pady=(14,0))
        tk.Label(hdr, text="◈  SONAR POINT CLOUD STUDIO",
                 bg=self.BG, fg=self.ACCENT,
                 font=("Courier New", 16, "bold")).pack(side='left')
        tk.Label(hdr, text="IMX 81A  ·  3D / 2D visualisation",
                 bg=self.BG, fg=self.SUBTEXT,
                 font=("Courier New", 9)).pack(side='left', padx=16)
        tk.Frame(self, bg=self.BORDER, height=1).pack(fill='x', padx=20)

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=20, pady=14)
        self._tab_3d = ttk.Frame(nb, style="TFrame")
        self._tab_2d = ttk.Frame(nb, style="TFrame")
        nb.add(self._tab_3d, text="  3D Point Cloud  ")
        nb.add(self._tab_2d, text="  2D Single Scan  ")
        self._build_3d_tab(self._tab_3d)
        self._build_2d_tab(self._tab_2d)

        self._status_var = tk.StringVar(value="Ready.")
        sb = tk.Frame(self, bg=self.PANEL, height=28)
        sb.pack(fill='x', side='bottom')
        tk.Label(sb, textvariable=self._status_var,
                 bg=self.PANEL, fg=self.SUBTEXT,
                 font=("Courier New", 9), anchor='w').pack(side='left', padx=10)
        self._prog = ttk.Progressbar(sb, style="Horizontal.TProgressbar",
                                     orient='horizontal', length=200, mode='determinate')
        self._prog.pack(side='right', padx=10, pady=4)

    # ══════════════════════════════════════════════════════════════════════════
    #  3D TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_3d_tab(self, tab):
        left  = ttk.Frame(tab); left.pack(side='left',  fill='both', expand=True)
        right = ttk.Frame(tab); right.pack(side='right', fill='y', padx=(0,6))

        src = self._section(left, "▸ Data Source")
        self._folder_row(src, ".81a folder", self._81a_folder,
            lambda: self._81a_folder.set(
                filedialog.askdirectory(title="Select .81a folder")
                or self._81a_folder.get()))
        self._folder_row(src, "CSV folder", self._csv_folder,
            lambda: self._csv_folder.set(
                filedialog.askdirectory(title="Select CSV folder")
                or self._csv_folder.get()))

        dp = self._section(left, "▸ Data Analysis Parameters")
        self._3d_start_az  = self._field(dp, "Start azimuth (°)",  "0")
        self._3d_step_az   = self._field(dp, "Azimuth step (°)",   "1.8")
        self._3d_threshold = self._field(dp, "Intensity threshold", "5")
        self._3d_ftype     = self._combo(dp, "Filter type",
                                          ["None","lowpass","highpass","bandpass"],
                                          "bandpass")
        self._3d_fsize     = self._field(dp, "Filter size", "1")

        lim = self._section(left, "▸ Plot Limits  (leave blank = no limit)")
        self._3d_xlim_lo = self._field(lim, "X min", "")
        self._3d_xlim_hi = self._field(lim, "X max", "")
        self._3d_ylim_lo = self._field(lim, "Y min", "")
        self._3d_ylim_hi = self._field(lim, "Y max", "")
        self._3d_zlim_lo = self._field(lim, "Z min", "")
        self._3d_zlim_hi = self._field(lim, "Z max", "")

        # right column
        pp = self._section(right, "▸ Plot Parameters")
        self._3d_colorby = self._combo(pp, "Color by", ["z","intensity"], "z")
        self._3d_cscale  = self._combo(pp, "Color scale",
                                        ["Viridis","custom","Plasma","Inferno",
                                         "Magma","Cividis","Turbo","Hot","Jet"],
                                        "Viridis")
        ttk.Label(pp, text="Custom colors (one per line,\nmatplotlib names):",
                  style="Sub.TLabel").pack(anchor='w', pady=(6,2))
        self._3d_custom_colors = tk.Text(
            pp, width=18, height=10,
            bg=self.ENTRY_BG, fg=self.TEXT, insertbackground=self.TEXT,
            relief='flat', font=("Courier New", 9),
            highlightbackground=self.BORDER, highlightthickness=1)
        self._3d_custom_colors.pack(fill='x')
        self._3d_custom_colors.insert('1.0',
            "indigo\nnavy\nmediumblue\nblue\ndarkcyan\nteal\n"
            "seagreen\nmediumseagreen\nlime\nchartreuse\nyellow")
        ttk.Label(pp, text="(only used when Color scale = custom)",
                  style="Sub.TLabel").pack(anchor='w', pady=(2,8))

        self._3d_diag = self._check(pp, "Print diagnostics", False)

        # PLY row with name field
        ply_row = ttk.Frame(pp, style="Panel.TFrame")
        ply_row.pack(fill='x', pady=2)
        self._3d_ply = tk.BooleanVar(value=False)
        ttk.Checkbutton(ply_row, text="Export .ply", variable=self._3d_ply,
                        style="TCheckbutton").pack(side='left')
        self._3d_ply_name = tk.StringVar(value="point_cloud")
        ttk.Entry(ply_row, textvariable=self._3d_ply_name,
                  width=13).pack(side='left', padx=(6,2))
        ttk.Label(ply_row, text=".ply", style="Sub.TLabel").pack(side='left')

        ttk.Separator(right, orient='horizontal').pack(fill='x', pady=8)
        ttk.Button(right, text="⟳  Convert .81a → CSV",
                   style="Muted.TButton", command=self._run_convert).pack(fill='x', pady=3)
        ttk.Button(right, text="▶  Build & Plot 3D",
                   style="Accent.TButton", command=self._run_3d).pack(fill='x', pady=3)

        do = self._section(left, "▸ Diagnostics Output")
        self._diag_text = tk.Text(
            do, height=10, bg=self.ENTRY_BG, fg=self.ACCENT2,
            insertbackground=self.TEXT, relief='flat', font=("Courier New", 9),
            highlightbackground=self.BORDER, highlightthickness=1)
        self._diag_text.pack(fill='both', expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  2D TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_2d_tab(self, tab):
        left  = ttk.Frame(tab); left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(tab); right.pack(side='right', fill='y', padx=(0,6))

        src = self._section(left, "▸ Select File to Plot")

        # browse .81a
        row1 = ttk.Frame(src, style="Panel.TFrame"); row1.pack(fill='x', pady=3)
        ttk.Label(row1, text=".81a file (auto-converts)",
                  style="Sub.TLabel", width=26).pack(side='left')
        self._2d_81a_path = tk.StringVar()
        ttk.Entry(row1, textvariable=self._2d_81a_path, width=32).pack(side='left', padx=4)
        ttk.Button(row1, text="Browse…", style="Muted.TButton",
                   command=self._browse_81a_2d).pack(side='left')

        # browse CSV
        row2 = ttk.Frame(src, style="Panel.TFrame"); row2.pack(fill='x', pady=3)
        ttk.Label(row2, text="CSV file",
                  style="Sub.TLabel", width=26).pack(side='left')
        self._2d_csv_path = tk.StringVar()
        ttk.Entry(row2, textvariable=self._2d_csv_path, width=32).pack(side='left', padx=4)
        ttk.Button(row2, text="Browse…", style="Muted.TButton",
                   command=self._browse_csv_2d).pack(side='left')

        ttk.Label(src,
                  text="— or pick from session (files converted / browsed this run) —",
                  style="Sub.TLabel").pack(anchor='w', pady=(10,2))

        lb_frame = ttk.Frame(src, style="Panel.TFrame")
        lb_frame.pack(fill='x')
        sb = tk.Scrollbar(lb_frame, orient='vertical', bg=self.PANEL)
        self._2d_listbox = tk.Listbox(
            lb_frame, bg=self.ENTRY_BG, fg=self.TEXT,
            selectbackground=self.ACCENT, selectforeground="#0d1117",
            font=("Courier New", 9), height=8, relief='flat',
            highlightbackground=self.BORDER, highlightthickness=1,
            yscrollcommand=sb.set)
        sb.config(command=self._2d_listbox.yview)
        self._2d_listbox.pack(side='left', fill='x', expand=True)
        sb.pack(side='right', fill='y')
        self._2d_listbox.bind('<<ListboxSelect>>', self._on_listbox_select)

        # plot params
        pp = self._section(right, "▸ Plot Parameters")
        self._2d_cscale = self._combo(pp, "Color scale",
                                       ["viridis","plasma","inferno","magma",
                                        "cividis","turbo","hot","jet"], "viridis")

        # save plot
        sp = self._section(right, "▸ Save Plot")
        self._2d_save     = self._check(sp, "Save plot to file", False)
        self._2d_savename = self._field(sp, "Filename (no ext)", "scan_plot", width=14)
        self._2d_savefmt  = self._combo(sp, "Format",
                                         ["jpg","png","pdf","svg","tiff"], "jpg")

        ttk.Separator(right, orient='horizontal').pack(fill='x', pady=8)
        ttk.Button(right, text="▶  Plot 2D Scan",
                   style="Green.TButton", command=self._run_2d).pack(fill='x', pady=3)

    # ── internal helpers ───────────────────────────────────────────────────────
    def _status(self, msg):
        self._status_var.set(msg)
        self.update_idletasks()

    def _set_progress(self, val, maximum=100):
        self._prog.configure(value=val, maximum=maximum)
        self.update_idletasks()

    def _log_diag(self, text):
        self._diag_text.delete('1.0', 'end')
        self._diag_text.insert('1.0', text)

    def _parse_lim(self, lo_var, hi_var):
        lo = lo_var.get().strip()
        hi = hi_var.get().strip()
        if lo == "" or hi == "":
            return None
        try:
            return (float(lo), float(hi))
        except ValueError:
            return None

    def _parse_colorscale(self):
        choice = self._3d_cscale.get()
        if choice != "custom":
            return choice
        raw    = self._3d_custom_colors.get('1.0', 'end').strip().splitlines()
        colors = [c.strip() for c in raw if c.strip()]
        bad    = [c for c in colors if not self._valid_color(c)]
        if bad:
            messagebox.showerror("Invalid colors",
                "These are not valid matplotlib color names:\n" + "\n".join(bad))
            return None
        if len(colors) < 2:
            messagebox.showerror("Too few colors", "Please enter at least 2 colors.")
            return None
        n = len(colors) - 1
        return [[i / n, c] for i, c in enumerate(colors)]

    @staticmethod
    def _valid_color(name):
        try:
            mcolors.to_rgba(name)
            return True
        except ValueError:
            return False

    def _add_to_session_list(self, csv_path, display_label=None):
        """Add csv_path to the 2D listbox if not already present."""
        label = display_label or os.path.basename(csv_path)
        existing = [self._2d_listbox.get(i)
                    for i in range(self._2d_listbox.size())]
        if label not in existing:
            self._2d_listbox.insert('end', label)
            self._listbox_paths.append(csv_path)

    def _browse_81a_2d(self):
        p = filedialog.askopenfilename(
            title="Select .81a file",
            filetypes=[(".81a files","*.81a"),("All","*.*")])
        if p:
            self._2d_81a_path.set(p)
            self._2d_csv_path.set("")

    def _browse_csv_2d(self):
        p = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files","*.csv"),("All","*.*")])
        if p:
            self._2d_csv_path.set(p)
            self._2d_81a_path.set("")

    def _on_listbox_select(self, _event):
        sel = self._2d_listbox.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self._listbox_paths):
                self._2d_csv_path.set(self._listbox_paths[idx])
                self._2d_81a_path.set("")

    def _populate_listbox(self):
        """Called after a 3D conversion — adds all resulting CSVs to session list."""
        for csv_path, az in self._converted_csvs:
            label = f"{os.path.basename(csv_path)}  (az {az:.1f}°)"
            self._add_to_session_list(csv_path, display_label=label)

    def _collect_scan_files(self):
        folder = self._81a_folder.get().strip()
        if not folder:
            messagebox.showerror("No folder", "Please select a .81a folder first.")
            return None, None
        files = sorted(glob.glob(os.path.join(folder, "*.81a")),
                       key=natural_sort_key)
        if not files:
            messagebox.showerror("No files", f"No .81a files found in:\n{folder}")
            return None, None
        start    = float(self._3d_start_az.get() or 0)
        step_deg = float(self._3d_step_az.get() or 1.8)
        azimuths = [start + i * step_deg for i in range(len(files))]
        return files, azimuths

    # ── actions ────────────────────────────────────────────────────────────────
    def _run_convert(self):
        files, azimuths = self._collect_scan_files()
        if files is None:
            return
        out_dir = self._csv_folder.get().strip()
        if not out_dir:
            out_dir = os.path.join(self._81a_folder.get().strip(), "csv_output")
        os.makedirs(out_dir, exist_ok=True)

        self._converted_csvs = []
        total = len(files)
        self._set_progress(0, total)

        for i, (fpath, az) in enumerate(zip(files, azimuths)):
            base     = os.path.splitext(os.path.basename(fpath))[0]
            csv_path = os.path.join(out_dir, base + ".csv")
            self._status(f"Converting {i+1}/{total}: {os.path.basename(fpath)}")
            self._set_progress(i, total)
            if not os.path.exists(csv_path):
                convert_81A_to_csv(fpath, csv_path)
            self._converted_csvs.append((csv_path, az))

        self._set_progress(total, total)
        self._populate_listbox()
        self._status(f"✓ Converted {total} file(s) → {out_dir}")
        messagebox.showinfo("Done",
            f"Converted {total} file(s).\nCSVs saved to:\n{out_dir}")

    def _run_3d(self):
        if self._converted_csvs:
            csv_files_with_azimuths = self._converted_csvs
        else:
            csv_folder = self._csv_folder.get().strip()
            if csv_folder and os.path.isdir(csv_folder):
                files    = sorted(glob.glob(os.path.join(csv_folder, "*.csv")),
                                  key=natural_sort_key)
                start    = float(self._3d_start_az.get() or 0)
                step_deg = float(self._3d_step_az.get() or 1.8)
                csv_files_with_azimuths = [
                    (f, start + i * step_deg) for i, f in enumerate(files)]
            else:
                messagebox.showerror("No data",
                    "Please convert .81a files first, or specify a CSV folder.")
                return

        try:
            threshold = float(self._3d_threshold.get() or 5)
            fsize     = int(self._3d_fsize.get() or 1)
        except ValueError:
            messagebox.showerror("Bad input",
                "Threshold and filter size must be numbers.")
            return

        ftype = self._3d_ftype.get()
        if ftype == "None":
            ftype = None

        xlim = self._parse_lim(self._3d_xlim_lo, self._3d_xlim_hi)
        ylim = self._parse_lim(self._3d_ylim_lo, self._3d_ylim_hi)
        zlim = self._parse_lim(self._3d_zlim_lo, self._3d_zlim_hi)

        colorscale = self._parse_colorscale()
        if colorscale is None:
            return

        total = len(csv_files_with_azimuths)

        def progress_cb(done, tot, name):
            self._status(f"Building cloud: {done}/{tot}  {name}")
            self._set_progress(done, tot)

        self._status("Building point cloud…")
        self._set_progress(0, total)

        points = build_point_cloud(
            csv_files_with_azimuths,
            intensity_threshold=threshold,
            filter_type=ftype,
            filter_size=fsize,
            xlim=xlim, ylim=ylim, zlim=zlim,
            progress_cb=progress_cb,
        )

        self._last_points = points
        self._set_progress(total, total)

        if self._3d_diag.get():
            self._log_diag(diagnostics_text(points))
        else:
            self._log_diag(f"Total points: {len(points):,}  "
                           f"(enable diagnostics for full report)")

        if len(points) == 0:
            self._status("⚠ No points — check filters / limits.")
            messagebox.showwarning("Empty cloud",
                "No points passed the filters.\n"
                "Try loosening threshold or limits.")
            return

        if self._3d_ply.get():
            name     = self._3d_ply_name.get().strip() or "point_cloud"
            ply_dir  = os.path.dirname(csv_files_with_azimuths[0][0])
            ply_path = os.path.join(ply_dir, name + ".ply")
            export_ply(points, ply_path)
            self._status(f"PLY saved → {ply_path}")

        self._status(f"✓ {len(points):,} points — opening browser…")
        plot_3d(points,
                color_by=self._3d_colorby.get(),
                colorscale=colorscale,
                title="360° Sonar Point Cloud")

    def _run_2d(self):
        # .81a takes priority over CSV field
        a81_path = self._2d_81a_path.get().strip()
        csv_path = None

        if a81_path:
            if not os.path.isfile(a81_path):
                messagebox.showerror("File not found", f"Cannot find:\n{a81_path}")
                return
            base     = os.path.splitext(a81_path)[0]
            csv_path = base + ".csv"
            if not os.path.exists(csv_path):
                self._status(f"Converting {os.path.basename(a81_path)}…")
                self.update_idletasks()
                convert_81A_to_csv(a81_path, csv_path)
            label = os.path.basename(csv_path) + "  (from .81a)"
            self._add_to_session_list(csv_path, display_label=label)
        else:
            csv_path = self._2d_csv_path.get().strip()
            if not csv_path or not os.path.isfile(csv_path):
                messagebox.showerror("No file",
                    "Please select a .81a or CSV file to plot.")
                return
            self._add_to_session_list(csv_path)

        # build save path if requested
        save_path = None
        if self._2d_save.get():
            name      = self._2d_savename.get().strip() or "scan_plot"
            fmt       = self._2d_savefmt.get()
            out_dir   = os.path.dirname(csv_path)
            save_path = os.path.join(out_dir, f"{name}.{fmt}")

        self._status(f"Plotting 2D: {os.path.basename(csv_path)}")
        try:
            plot_2d(csv_path,
                    colorscale=self._2d_cscale.get(),
                    save_path=save_path)
            msg = "✓ 2D plot displayed."
            if save_path:
                msg += f"  Saved → {save_path}"
            self._status(msg)
        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._status("Error during 2D plot.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SonarApp()
    app.mainloop()