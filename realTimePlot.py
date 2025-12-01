#!/usr/bin/env python3
import time
import os
from collections import deque

STATUSFILE = os.path.expanduser("~/TME160/bubbleColumn/simulation_status.txt")
LOGFILE = os.path.expanduser("~/TME160/bubbleColumn/simulation_log_display.txt")
MAX_LINES = 30

history = deque(maxlen=MAX_LINES)
header = f"{'t':>6} {'ID':>3} {'x':>7} {'y':>7} {'u':>7} {'v':>7} {'D':>7} {'Vrel':>7} {'Re':>7} {'Cd':>6} {'Fb':>8} {'Fdy':>8} {'FtotY':>8}"
separator = "=" * 95

last_line = ""
first_draw = True

while True:
    try:
        with open(STATUSFILE, "r") as f:
            line = f.read().strip()
            
        if line and line != last_line:
            last_line = line
            values = line.split(',')
            if len(values) >= 17:
                try:
                    t = float(values[0])
                    bID = int(values[1])
                    x = float(values[2])
                    y = float(values[3])
                    u = float(values[4])
                    v = float(values[5])
                    D = float(values[6])
                    Vrel = float(values[7])
                    Re = float(values[8])
                    Cd = float(values[9])
                    Fb = float(values[10])
                    Fdy = float(values[11])
                    FtotY = float(values[13])
                    
                    formatted_line = f"{t:6.2f} {bID:3d} {x:7.4f} {y:7.4f} {u:7.4f} {v:7.4f} {D:7.5f} {Vrel:7.4f} {Re:7.1f} {Cd:6.3f} {Fb:8.5f} {Fdy:8.5f} {FtotY:8.5f}"
                    
                    history.append(formatted_line)
                    
                    if first_draw:
                        print("\033[2J\033[H")
                        print(separator)
                        print(header)
                        print(separator)
                        first_draw = False
                    
                    print("\033[4;0H\033[J", end='')
                    for hist_line in history:
                        print(hist_line)
                    
                    with open(LOGFILE, 'w') as f:
                        f.write(separator + "\n")
                        f.write(header + "\n")
                        f.write(separator + "\n")
                        for hist_line in history:
                            f.write(hist_line + "\n")
                    
                except (ValueError, IndexError):
                    pass
        
        time.sleep(0.05)
    except FileNotFoundError:
        if first_draw:
            print("\033[2J\033[H")
            first_draw = False
        print("\033[4;0H\033[KWaiting for simulation_status.txt...")
        time.sleep(1)
