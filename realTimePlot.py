#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

LOGFILE = "simulation_log.csv"

plt.ion()
fig, ax = plt.subplots(3, 1, figsize=(8, 10))  # <-- now 3 rows

def follow(filename):
    """Generator that yields new lines as the file grows (like tail -f)"""
    with open(filename, "r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            yield line

# Storage
df = pd.DataFrame()

stream = follow(LOGFILE)

while True:
    line = next(stream)
    if line.startswith("time"):  # skip header
        continue

    values = line.strip().split(',')
    row = {
        "time": float(values[0]),
        "bubbleID": int(values[1]),
        "x": float(values[2]),
        "y": float(values[3]),
        "u": float(values[4]),
        "v": float(values[5]),
        "D": float(values[6]),
        "Vrel": float(values[7]),
        "Re": float(values[8]),
        "Cd": float(values[9]),
        "Fb": float(values[10]),
        "Fdy": float(values[11]),
        "Fhist": float(values[12]),
        "FtotY": float(values[13]),
        "FT": float(values[14]),
        "Fdx": float(values[15]),
        "FtotX": float(values[16]),
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Keep viewer fast
    if len(df) > 2000:
        df = df.iloc[-2000:]

    # --- Clear subplots ---
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    # --- Titles ---
    ax[0].set_title("Bubble vertical velocity (v) over time")
    ax[1].set_title("Forces over time (log scale)")
    ax[2].set_title("Cd over time")

    # --- Plot values for each bubble ---
    for bubble in df["bubbleID"].unique():
        d = df[df["bubbleID"] == bubble]

        # Velocity
        ax[0].plot(d["time"], d["v"], label=f"Bubble {bubble}")

        # All forces (same as before)
        ax[1].plot(d["time"], d["Fb"],    label=f"Fb (bubble {bubble})")
        ax[1].plot(d["time"], d["Fdy"],   label=f"Fdy (bubble {bubble})")
        ax[1].plot(d["time"], d["Fhist"], label=f"Fhist (bubble {bubble})")
        ax[1].plot(d["time"], d["FtotY"], label=f"FtotY (bubble {bubble})")
        ax[1].plot(d["time"], d["Fdx"],   label=f"Fdx (bubble {bubble})")
        ax[1].plot(d["time"], d["FtotX"], label=f"FtotX (bubble {bubble})")

        # Cd
        ax[2].plot(d["time"], d["Cd"], label=f"Bubble {bubble}")

    # Labels
    ax[0].set_xlabel("time [s]")
    ax[1].set_xlabel("time [s]")
    ax[2].set_xlabel("time [s]")

    ax[0].set_ylabel("v [m/s]")
    ax[1].set_ylabel("Force [N]")
    ax[2].set_ylabel("Cd [-]")

    # Log scale for forces
    ax[1].set_yscale("log")

    # Legends
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.pause(0.01)
