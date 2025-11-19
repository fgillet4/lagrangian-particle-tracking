#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 13:30:52 2025

@author: francisbrain4
"""



"""

Simulation of a spherical particle

Bastian, Francis

"""

import math
import numpy as np
import matplotlib.pyplot as plt

rho_f = 1000.0 # kg/m3
rho_p = 2560.0 # kg/m3
d_p = 0.5e-3# m
g = 9.81 # m/s2
mu_f = 1.0e-3

# Initial conditions
v_0 = 0.0 # m/s
y_0 = 0.0

max_iter = 200
tolerance = 1e-8
dt = 1e-4               # Time step (s)
t_max = 40              # Max. simulation time (s)

# Particle Mass Calculation
volume = math.pi / 6.0 * d_p**3       # Particle volume (m^3)
area = math.pi / 4.0 * d_p**2         # Projected area (m^2)
m_p = rho_p * volume                  # Particle mass (kg)

def drag_coefficient(Re):
    if Re < 0.1:
        return 24.0 / Re
    if Re <= 1000.0:
        return 24.0 / Re * (1.0 + 0.15 * Re**0.687)
    return 0.44

def drag_force(v, rho_f_local=rho_f, mu_local=mu_f):
    rel_v = v  # fluid velocity = 0
    Re = abs(rel_v) * rho_f_local * d_p / mu_local if abs(rel_v) > 0 else 0.0
    if Re == 0.0:
        Cd = 0.0
        Fd = 0.0
    else:
        Cd = drag_coefficient(Re)
        Fd_mag = 0.5 * rho_f_local * Cd * area * rel_v**2
        Fd = -math.copysign(Fd_mag, rel_v)  # opposite sign to rel_v
    return Fd, Re, Cd

def added_mass_force(v,m_local=m_p,rho_f_local=rho_f,rho_p_local=rho_p,dt_local=dt):
    Fam = 0.5*m_p*rho_f_local/rho_p_local*(0-v)/dt
    return Fam

def history_force(v, m_local=m_p, t_max_local=t_max):
    Fhist = m_p*(v-0/(math.sqrt(0.5*t_max_local)))
    return(Fhist)

nt = int(t_max / dt) + 1
time = np.zeros(nt)
vel = np.zeros(nt)
pos = np.zeros(nt)
acc = np.zeros(nt)
Re_arr = np.zeros(nt)
Cd_arr = np.zeros(nt)
Fd_arr = np.zeros(nt)
Fg_arr = np.zeros(nt)
Fb_arr = np.zeros(nt)
Fam_arr = np.zeros(nt)
Fhist_arr = np.zeros(nt)

# initial values
vel[0] = v_0
pos[0] = y_0

F_gravity = m_p * g
F_buoyancy = rho_f * volume * g
Fg_minus_b = F_gravity - F_buoyancy

converge_count = 0
last_index = nt - 1

for n in range(nt - 1):
    time[n] = n * dt
    # compute drag at current velocity
    Fd, Re, Cd = drag_force(vel[n])
    Fam =  added_mass_force(vel[n])
    Fhist =  history_force(vel[n])*0
    Fd_arr[n] = Fd
    Re_arr[n] = Re
    Cd_arr[n] = Cd
    Fg_arr[n] = F_gravity
    Fb_arr[n] = F_buoyancy
    Fam_arr[n] = Fam
    Fhist_arr[n] = Fhist
    
    # total force downward positive
    F_total = Fg_minus_b + Fd + Fam + Fhist
    a = F_total / m_p
    acc[n] = a

    # Forward Euler updates
    vel[n + 1] = vel[n] + dt * a
    pos[n + 1] = pos[n] + dt * vel[n]

    # check convergence to terminal velocity
    if abs(vel[n + 1] - vel[n]) < tolerance:
        converge_count += 1
    else:
        converge_count = 0

    if converge_count >= max_iter:
        # truncate arrays up to n+1
        last_index = n + 1
        time[last_index] = last_index * dt
        break

# if loop completed without break, ensure last time stored
if last_index == nt - 1:
    time[-1] = (nt - 1) * dt

# slice arrays to actual length
time = time[:last_index + 1]
vel = vel[:last_index + 1]
pos = pos[:last_index + 1]
acc = acc[:last_index + 1]
Re_arr = Re_arr[:last_index + 1]
Cd_arr = Cd_arr[:last_index + 1]
Fd_arr = Fd_arr[:last_index + 1]
Fg_arr = Fg_arr[:last_index + 1]
Fb_arr = Fb_arr[:last_index + 1]

data = [
(4.32147, 0.0199711),
(10.0679, 0.0315119),
(15.1502, 0.0398836),
(20.5745, 0.0483152),
(23.9632, 0.0538167),
(30.0853, 0.0602753),
(36.2227, 0.0644021),
(43.3926, 0.0677519),
(47.8333, 0.0694867),
(53.9895, 0.0707435),
(60.4889, 0.0718809),
(68.3601, 0.0727195),
(74.8627, 0.0733785),
(85.1335, 0.0738589),
(96.0901, 0.0742198),
(109.786, 0.0745813),
(120.402, 0.0746432),
(133.758, 0.0747654),
(142.662, 0.0747672),
(158.073, 0.0747703),
(174.854, 0.0747138),
(190.607, 0.074717),
(204.648, 0.0747198),
(219.718, 0.0746032),
(241.293, 0.0745477),
(256.021, 0.0743713),
(267.665, 0.074254),
(281.364, 0.0742567),
(292.322, 0.0743785),
(306.364, 0.0742617),
(324.514, 0.0742653),
(344.377, 0.0743889),
(360.473, 0.0743323),
(376.569, 0.0743355),
(390.952, 0.0743982),
(399.858, 0.0741608)
]

time_ms_data = [row[0] for row in data]
velocity_data = [row[1] for row in data]

# Plot 1: Particle Velocity vs Time
plt.figure(figsize=(8,4))
plt.plot(time*1000, vel, 'b-', linewidth=2)
plt.plot(time_ms_data, velocity_data, 'r--')
plt.xlabel("Time (ms)")
plt.ylabel("Particle velocity (m/s)")
plt.title("Particle settling velocity vs time (Forward Euler)")
plt.grid(True)
plt.tight_layout()

# Plot 2: Reynolds Number vs Time
plt.figure(figsize=(8,4))
plt.plot(time*1000, Re_arr, 'r-', linewidth=2)
plt.xlabel('Time (ms)', fontsize=11)
plt.ylabel('Reynolds Number', fontsize=11)
plt.title('Particle Reynolds Number vs Time', fontsize=12, fontweight='bold')
plt.grid(True)
plt.tight_layout()

plt.savefig('particle_simulation_results.png', dpi=300, bbox_inches='tight')
plt.show()