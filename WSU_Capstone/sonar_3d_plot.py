import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d
import os
import glob
import re
import tkinter as tk
from tkinter import filedialog

# --- Constants ---
SHOT_SIZE = 384
ECHO_BYTES = 252


##----------------------------------------------------------------------------------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------##

START_AZIMUTH_DEG = 0                      ## azimuth of the first file (degrees)
AZIMUTH_STEP_DEG  = 1.8                    ## degrees between each scan

# Set to an integer N to use every Nth file (e.g. 2 = every other file),
# or None to use all files in the folder.
FILE_STEP = None

INTENSITY_THRESHOLD  = 5                   ## raise to suppress more noise (post-filter)
FILTER_TYPE          = 'bandpass'          ## None | 'lowpass' | 'highpass' | 'bandpass'
FILTER_SIZE          = 1                   ## kernel size — larger = more aggressive
COLOR_SCALE          = 'z'                 ## z == colorbar based on height | intensity == colorbar bar based on signel strength

##----------------------------------------------------------------------------------------------------------------------------##

XLIM = (-4.5,8)
YLIM = (-0.75,3.3)
ZLIM = (-3,0)

##----------------------------------------------------------------------------------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------##




# --- Helpers ---
def decode_head_position(b5, b6):
    hi = (b6 & 0x3E) >> 1
    lo = ((b6 & 0x01) << 7) | (b5 & 0x7F)
    return (hi << 8) | lo

def decode_profile_range(b8, b9):
    hi = (b9 & 0x7E) >> 1
    lo = ((b9 & 0x01) << 7) | (b8 & 0x7F)
    return (hi << 8) | lo

# --- Natural sort ---
def natural_sort_key(path):
    name = os.path.basename(path)
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

# --- Parser ---
def convert_81A_to_csv(input_file, output_csv):
    
    rows = []
    
    with open(input_file, "rb") as fh:
        while True:
            shot = fh.read(SHOT_SIZE)
            
            if len(shot) < SHOT_SIZE:
                break
                
            shot = list(shot)
            id_str    = bytes(shot[0:3]).decode("ascii", errors="ignore")
            shotBytes = (shot[4] << 8) | shot[5]
            nToRead   = (shot[6] << 8) | shot[7]
            date  = bytes(shot[8:20]).decode("ascii", errors="ignore")
            time  = bytes(shot[20:29]).decode("ascii", errors="ignore")
            hund  = bytes(shot[29:33]).decode("ascii", errors="ignore")
            rh = shot[100:116]
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
                "ProfileUnits": profUnits, "Profile_m": round(prof_m, 4)
            }
            
            for i in range(ECHO_BYTES):
                row[f"Echo_{i}"] = shot[112 + i]
                
            rows.append(row)
            
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df

# --- Filter ---
def apply_echo_filter(echo_array, filter_type=None, filter_size=5):
    """
    filter_type:
        None        — no filtering
        'lowpass'   — smooths echo, suppresses high-freq speckle noise
        'highpass'  — keeps sharp returns, removes slow background drift
        'bandpass'  — combination of both (recommended)
    filter_size: kernel size (larger = more aggressive)
    """
    if filter_type is None:
        return echo_array.astype(float)
    
    
    smoothed = uniform_filter1d(echo_array.astype(float), size=filter_size)
    
    
    if filter_type == 'lowpass':
        return smoothed
    
    elif filter_type == 'highpass':
        return echo_array.astype(float) - smoothed
    
    elif filter_type == 'bandpass':
        lp = uniform_filter1d(echo_array.astype(float), size = filter_size)
        hp = lp - uniform_filter1d(lp, size = filter_size * 3)
        return hp
    
    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'. Use None, 'lowpass', 'highpass', or 'bandpass'.")

# --- Point cloud builder ---
def build_point_cloud(csv_files_with_azimuths,
                      intensity_threshold = 10,
                      filter_type = None,
                      filter_size = 5,
                      xlim = None, ylim = None, zlim = None):
    all_points = []

    for csv_path, azimuth_deg in csv_files_with_azimuths:
        
        data = pd.read_csv(csv_path)
        echo_cols    = [f"Echo_{i}" for i in range(ECHO_BYTES)]
        echo_matrix  = data[echo_cols].values
        sonar_angles = data["Angle_deg"].values
        range_metres = data["Range_m"].values
        azimuth_rad  = np.deg2rad(azimuth_deg)

        for shot_idx, theta_deg in enumerate(sonar_angles):
            theta_rad  = np.deg2rad(theta_deg)
            intensities = apply_echo_filter(echo_matrix[shot_idx], filter_type, filter_size)

            # Scale r by the actual sonar range for real-world metres
            shot_range = range_metres[shot_idx]
            r_values   = np.linspace(0, shot_range, ECHO_BYTES)

            for r, intensity in zip(r_values, intensities):
                if intensity < intensity_threshold:
                    continue

                    
                horiz = r * np.sin(theta_rad)
                
                x = horiz * np.cos(azimuth_rad)
                y = -horiz * np.sin(azimuth_rad)
                z = r * np.cos(theta_rad)

                
                if xlim and not (xlim[0] <= x <= xlim[1]): continue
                if ylim and not (ylim[0] <= y <= ylim[1]): continue
                if zlim and not (zlim[0] <= z <= zlim[1]): continue

                all_points.append([x, y, z, intensity])

    return np.array(all_points)

def print_diagnostics(points):
    
    print(f"Total points: {len(points):,}")
    
    if len(points) > 0:
        
        print(f"X range: {points[:,0].min():.2f} to {points[:,0].max():.2f} m")
        print(f"Y range: {points[:,1].min():.2f} to {points[:,1].max():.2f} m")
        print(f"Z range: {points[:,2].min():.2f} to {points[:,2].max():.2f} m")
        
        
        print(f"Intensity range: {points[:,3].min():.1f} to {points[:,3].max():.1f}")
        bins = [0, 10, 20, 30, 50, 75, 100, 128]
        hist, edges = np.histogram(points[:,3], bins=bins)
        print("\nIntensity distribution:")
        
        
        for i in range(len(hist)):
            print(f"  {int(edges[i]):>3} - {int(edges[i+1]):>3}: {hist[i]:>10,} points")
            
    else:
        print("WARNING: No points passed the filters!")
        print("  Try setting INTENSITY_THRESHOLD = 0, FILTER_TYPE = None, all limits to None.")


# --- Interactive plot ---
def plot_point_cloud_interactive(points, title="360° Sonar Point Cloud"):
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    
    color_values = z if COLOR_SCALE == 'z' or COLOR_SCALE == 'Z' else intensity
    color_label  = 'Z Depth [m]' if COLOR_SCALE == 'z' else 'Intensity [NA]'

    colorscale = [
            [0.0, "indigo"],      # deepest
            [0.1, "navy"],
            [0.2, "mediumblue"],
            [0.3, "blue"],
            [0.4, "darkcyan"],
            [0.5, "teal"],
            [0.6, "seagreen"],
            [0.7, "mediumseagreen"],
            [0.8, "lime"],
            [0.9, "chartreuse"],
            [1.0, "yellow"]       # shallow
        ]     
    
    fig = go.Figure(data = [go.Scatter3d(
        x = x, y = y, z = z,
        
        mode='markers',
        marker = dict(
            size = 1.5,
            color = color_values,
            colorscale = colorscale, #  'Viridis',
            colorbar = dict(title = color_label),
            opacity = 1,
        )
    )])

    fig.update_layout(
        title = title,
        scene = dict(
            xaxis_title = 'X [m]',
            yaxis_title = 'Y [m]',
            zaxis_title = 'Z [m]',
            aspectmode ='data',
        ),
        margin = dict(l = 0, r = 0, b = 0, t = 40),
    )

    fig.show()

# --- Folder picker ---
def pick_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title = "Select folder containing .81a scan files")
    root.destroy()
    return folder

# --- File step filter ---
def filter_files_by_step(scan_files, step=None):
    """
    step : None — use all files
           int N — take every Nth file (e.g. 2 = every other file)
    """
    if step is None:
        return scan_files
    
    return scan_files[::step]

def export_ply(points, filepath):
    """Export point cloud to PLY format."""
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float intensity\n")
        f.write("end_header\n")
        # Data
        for xi, yi, zi, ii in zip(x, y, z, intensity):
            f.write(f"{xi:.6f} {yi:.6f} {zi:.6f} {ii:.6f}\n")
    
    print(f"PLY exported to: {filepath}")


'''##########################################################################################################################'''

                                                  #######   ###   #  ##    #
                                                  #  #  #  #   #     # #   #
                                                  #  #  #  #####  #  #  #  #
                                                  #  #  #  #   #  #  #   # #
                                                  #     #  #   #  #  #    ##

'''##########################################################################################################################'''


if __name__ == "__main__":

    # Step 1: Pick folder
    scan_folder = pick_folder()
    if not scan_folder:
        print("No folder selected. Exiting.")
        exit()
        

                    ##----------- Step 2: Find, natural-sort, and optionally thin the file list -----------##

    scan_files = filter_files_by_step(
        sorted(glob.glob(os.path.join(scan_folder, "*.81a")), key=natural_sort_key),
        FILE_STEP
    )

    if not scan_files:
        print(f"No .81a files found in: {scan_folder}")
        exit()

    n_files = len(scan_files)
    print(f"\nFound {n_files} scan file(s) in: {scan_folder}")
    
    
                    ##----------------------------- Step 3: Assign azimuths ------------------------------##

    azimuths = [START_AZIMUTH_DEG + i * AZIMUTH_STEP_DEG for i in range(n_files)]

    '''print("\nFile → Azimuth mapping:")
    for f, az in zip(scan_files, azimuths):
        print(f"  {os.path.basename(f):40s} → {az:.2f}°")'''
        
    print(f"\nTotal sweep: {azimuths[-1]:.2f}° "
          f"(files: {n_files}, step: {AZIMUTH_STEP_DEG}°)")
    
    
#                                                                                                                              #
##----------------------------------------------- Step 4: Convert .81a to CSV ------------------------------------------------##
#                                                                                                                              #

    csv_dir = os.path.join(scan_folder, "csv_output")
    os.makedirs(csv_dir, exist_ok = True)

    csv_files_with_azimuths = []
    
    for input_path, azimuth in zip(scan_files, azimuths):
        
        base     = os.path.splitext(os.path.basename(input_path))[0]
        csv_path = os.path.join(csv_dir, base + ".csv")
        
        print(f"Converting {os.path.basename(input_path)} ...")
        
        convert_81A_to_csv(input_path, csv_path)
        csv_files_with_azimuths.append((csv_path, azimuth))
        
        
#                                                                                                                              #
##------------------------------------------------ Step 5: Build point cloud -------------------------------------------------##
#                                                                                                                              #
    
    print("\nBuilding point cloud...")
    
    points = build_point_cloud(
        csv_files_with_azimuths,
        intensity_threshold=INTENSITY_THRESHOLD,
        filter_type = FILTER_TYPE,
        filter_size = FILTER_SIZE,
        
        xlim = XLIM,
        ylim = YLIM,
        zlim = ZLIM,
    )
    
    # Export PLY
    ply_path = os.path.join(scan_folder, "point_cloud_full_pool.ply")
    export_ply(points, ply_path)

#                                                                                                                              #
##---------------------------------------------------- Step 6: Diagnostics ---------------------------------------------------##
    """
    This diagnostic function is to identify any issue with the plot. It is not necessary for the code to function
    """
    print_diagnostics(points)

    
#                                                                                                                              #
##------------------------------------------------------- Step 7: Plot -------------------------------------------------------##
#                                                                                                                              #
    if len(points) > 0:
        plot_point_cloud_interactive(points)
    else:
        print("Nothing to plot — check your filter settings.")