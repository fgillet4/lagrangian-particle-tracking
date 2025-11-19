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

global rho_fluid, rho_particle, particle_dia, mu_fluid, g, mp # kg/m3
rho_fluid = 1000
rho_particle = 2560 # kg/m3
particle_dia = 0.5e-3# mm
g = 9.81 # m/s2

mp = (math.pi/6) * particle_dia**3 * rho_particle

mu_fluid = 0.0008891

vel_fluid = 0 # m/s


max_iter = 400000 
tolerance = 1e-8
step_size = 1e-6


vel_particle = []
Re_p_list = []
dvdt = []
t = []
position = []


grav_force_list = []
hist_force_list= []
added_mass_force_list = []
drag_force_list = []
buyancy_force_list = []
total_force_list = []


class solid_spherical_particle:
    # Class-level lookup table for drag crisis region (shared by ALL particles)
    # Digitized from drag coefficient vs Reynolds number curve
    Re_drag_crisis = np.array([
        10472.3881,
        18778.4117,
        30977.1558,
        47010.3815,
        67015.178,
        108267.5342,
        160913.6439,
        239159.4209,
        313643.6117,
        355452.945,
        370593.1097,
        386378.1546,
        402835.5478,
        598718.1323,
        1008469.3544,
        1663586.5015,
        2802113.2392,
        3831332.6321,
        5576823.1232
    ])
    
    Cd_drag_crisis = np.array([
        0.3996,
        0.4471,
        0.491,
        0.5097,
        0.5493,
        0.5703,
        0.5597,
        0.5391,
        0.4819,
        0.3639,
        0.1999,
        0.1184,
        0.0829,
        0.0814,
        0.1,
        0.13,
        0.1627,
        0.1855,
        0.2037
    ])
    def __init__(self, diameter, density, initial_position, initial_velocity ): # these are the variables we need to create a particle!!
        # This is where we define the state variables
        self.position = initial_position
        self.velocity = initial_velocity
        self.time = 0
        self.diameter = diameter #meters
        self.density = density #kg/m3
        self.mass = (1/6)*math.pi*diameter**3 #kg
    
    def calc_drag(self, rho_fluid, mu_fluid, vel_fluid):
        # Correct area calculation
        area = math.pi * self.diameter**2 / 4
        Re_p = self.calc_Re_p(rho_fluid, vel_fluid,mu_fluid)
        if Re_p < 0.01:
            return 3*math.pi*mu_fluid*self.diameter*(vel_fluid-self.velocity)
        elif Re_p < 1:
            Cd = 24/Re_p
        elif Re_p < 1000:
            Cd = (24/Re_p)*(1+0.15*Re_p**(0.687))
        elif Re_p < 10472.3881:  # Below drag crisis data range
            # Newton regime
            Cd = 0.44
        elif Re_p <= 5576823.1232:  # Within drag crisis data range
            # Use lookup table with logarithmic interpolation
            Cd = np.interp(np.log10(Re_p), 
                          np.log10(self.Re_drag_crisis), 
                          self.Cd_drag_crisis)
        else:  # Above drag crisis data range
            # Post-crisis regime
            Cd = 0.19
        # Correct drag force
        Fd = 0.5 * rho_fluid * area * Cd * abs(vel_fluid-self.velocity) * (vel_fluid-self.velocity)
        return Fd

    def calc_buoyancy(self, rho_f, g=9.81):
        """Calculate buoyancy force"""
        V_p = self.mass / self.density  # Particle volume
        Fb = -rho_f * V_p * g  # Negative because upward
        return Fb

    def gravity_force(self,g=9.81):
        Fg = self.mass*g    
        return Fg
    
    def calc_total_force(self, rho_fluid, mu_fluid, vel_fluid, g=9.81):
        Fd = self.calc_drag(rho_fluid, mu_fluid, vel_fluid)
        Fb = self.calc_bouyancy(rho_fluid, g)
        Fg = self.gravity_force(g)
        return Fd + Fb + Fg

    def update_velocity(self, total_force, dt):
        acceleration = total_force / self.mass
        self.velocity = self.velocity + acceleration * dt

    def update_position(self, dt):
        self.position = self.position + self.velocity * dt

    def calc_Re_p(self, rho_fluid, vel_fluid,mu_fluid):
        Re_p = rho_fluid*abs(vel_fluid-self.velocity)*self.diameter/mu_fluid
        return (Re_p)

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.position









def Re_p(rho_fluid, vel_fluid, vel_particle, dia_particle, mu_fluid):
    
    Re_p = rho_fluid*abs(vel_fluid-vel_particle)*dia_particle/mu_fluid
    
    return (Re_p)



def drag_force(Re_p, rho_fluid, dia_particle, vel_fluid, vel_particle, mu_fluid):
    # Correct area calculation
    area = math.pi * dia_particle**2 / 4
    
    if Re_p < 0.01:
        return 3*math.pi*mu_fluid*dia_particle*(vel_fluid-vel_particle)
    elif Re_p < 1:
        Cd = 24/Re_p
    else:  # Re_p >= 1
        Cd = (24/Re_p)*(1+0.15*Re_p**(0.687))
    
    # Correct drag force
    Fd = 0.5 * rho_fluid * area * Cd * abs(vel_fluid-vel_particle) * (vel_fluid-vel_particle)
    return Fd


def added_mass_force(m_p,rho_f,rho_p,vel_fluid,vel_particle,dt):
    Fam = 0.5*(m_p*rho_fluid/rho_particle)*(vel_fluid-vel_particle)/dt
    
    return Fam


def gravity_force(m,g=9.81):
    Fg = m*g    
    return Fg

def history_force(m_p,vel_particle, max_iter, dt):
    
    T = max_iter*dt
    
    Fhist = m_p*(vel_particle-0/(math.sqrt(0.5*T)))

    return(Fhist)

def buoyancy_force(m, rho_f, rho_p, g=9.81):
    """Calculate buoyancy force"""
    V_p = m / rho_p  # Particle volume
    Fb = -rho_f * V_p * g  # Negative because upward
    return Fb

def Ftot(Fd,Fam,Fg,Fhist,Fb):
    
    return(Fd + Fam*0 + Fg + Fhist*0 + Fb)

vel_particle.append(0.0000001)
t.append(0)
position.append(0) # starting position at zero meters


for i in range(1,max_iter):
    
    Re_p_temp = Re_p(rho_fluid, vel_fluid, vel_particle[i-1], particle_dia, mu_fluid)
    Re_p_list.append(Re_p_temp)
    Fd = drag_force(Re_p_temp, rho_fluid, particle_dia, vel_fluid, vel_particle[i-1], mu_fluid)
    drag_force_list.append(Fd)
    Fam = added_mass_force(mp, rho_fluid, rho_particle, vel_fluid, vel_particle[i-1], step_size)
    added_mass_force_list.append(Fam)
    Fg = gravity_force(mp)
    grav_force_list.append(Fg)
    Fhist = history_force(mp, vel_particle[i-1], max_iter, step_size)
    hist_force_list.append(Fhist)
    Fb = buoyancy_force(mp,rho_fluid, rho_particle)
    buyancy_force_list.append(Fb)
    Ftot_temp = float(Ftot(Fd,Fam,Fg,Fhist,Fb))
    total_force_list.append(Ftot_temp)
    
    vel_particle.append(vel_particle[i-1]+Ftot_temp*step_size/mp)
    position.append(position[i-1] + vel_particle[i-1] * step_size)
    t.append(t[i-1]+step_size)
    #if (abs(vel_particle[i] - vel_particle[i-1])) < tolerance:
    #    break

import matplotlib.pyplot as plt

# Convert time to milliseconds for plotting
t_ms = [ti * 1000 for ti in t]
t_ms_shifted = [ti * 1000 for ti in t[1:]]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Particle Velocity vs Time
axes[0, 0].plot(t_ms, vel_particle, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time (ms)', fontsize=11)
axes[0, 0].set_ylabel('Particle Velocity (m/s)', fontsize=11)
axes[0, 0].set_title('Particle Velocity vs Time', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Reynolds Number vs Time
axes[0, 1].plot(t_ms_shifted, Re_p_list, 'r-', linewidth=2)
axes[0, 1].set_xlabel('Time (ms)', fontsize=11)
axes[0, 1].set_ylabel('Reynolds Number', fontsize=11)
axes[0, 1].set_title('Particle Reynolds Number vs Time', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: All Forces vs Time
axes[1, 0].plot(t_ms_shifted, drag_force_list, label='Drag Force', linewidth=2)
axes[1, 0].plot(t_ms_shifted, added_mass_force_list, label='Added Mass Force', linewidth=2)
axes[1, 0].plot(t_ms_shifted, grav_force_list, label='Gravity Force', linewidth=2)
axes[1, 0].plot(t_ms_shifted, hist_force_list, label='History Force', linewidth=2)
axes[1, 0].plot(t_ms_shifted, buyancy_force_list, label='Buoyancy Force', linewidth=2)
axes[1, 0].set_xlabel('Time (ms)', fontsize=11)
axes[1, 0].set_ylabel('Force (N)', fontsize=11)
axes[1, 0].set_title('Individual Forces vs Time', fontsize=12, fontweight='bold')
axes[1, 0].legend(loc='best', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Total Force vs Time
axes[1, 1].plot(t_ms_shifted, total_force_list, 'k-', linewidth=2)
axes[1, 1].set_xlabel('Time (ms)', fontsize=11)
axes[1, 1].set_ylabel('Total Force (N)', fontsize=11)
axes[1, 1].set_title('Total Force vs Time', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('particle_simulation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Create a separate plot for acceleration
fig2, ax = plt.subplots(figsize=(10, 6))
acceleration = [F/mp for F in total_force_list]
ax.plot(t_ms_shifted, acceleration, 'g-', linewidth=2)
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Particle Acceleration (m/s²)', fontsize=12)
ax.set_title('Particle Acceleration vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('particle_acceleration.png', dpi=300, bbox_inches='tight')
plt.show()

fig3, ax = plt.subplots(figsize=(10, 6))
position_mm = [p * 1000 for p in position]  # Convert to mm for easier reading
ax.plot(t_ms, position_mm, 'purple', linewidth=2)
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Particle Position (mm)', fontsize=12)
ax.set_title('Particle Position vs Time (Falling Distance)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('particle_position.png', dpi=300, bbox_inches='tight')
plt.show()