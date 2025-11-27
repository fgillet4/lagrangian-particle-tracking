import numpy as np
import math


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
        self.position = np.atleast_1d(np.array(initial_position, dtype=float))
        self.velocity = np.atleast_1d(np.array(initial_velocity, dtype=float))
        self.time = 0
        self.diameter = diameter #meters
        self.density = density #kg/m3
        self.mass = (1/6)*math.pi*diameter**3*density #kg
    
    def calc_drag(self, rho_fluid, mu_fluid, vel_fluid):
        area = math.pi * self.diameter**2 / 4
        vel_fluid = np.atleast_1d(np.array(vel_fluid, dtype=float))
        
        vel_rel = vel_fluid - self.velocity
        vel_rel_mag = np.linalg.norm(vel_rel)
        
        Re_p = self.calc_Re_p(rho_fluid, vel_fluid, mu_fluid)
        
        if Re_p < 0.01:
            return 3*math.pi*mu_fluid*self.diameter*vel_rel
        elif Re_p < 1:
            Cd = 24/Re_p
        elif Re_p < 1000:
            Cd = (24/Re_p)*(1+0.15*Re_p**(0.687))
        elif Re_p < 10472.3881:
            Cd = 0.44
        elif Re_p <= 5576823.1232:
            Cd = np.interp(np.log10(Re_p), 
                          np.log10(self.Re_drag_crisis), 
                          self.Cd_drag_crisis)
        else:
            Cd = 0.19
        
        if vel_rel_mag > 0:
            Fd = 0.5 * rho_fluid * area * Cd * vel_rel_mag * vel_rel
        else:
            Fd = np.zeros_like(vel_rel)
        
        return Fd

    def calc_buoyancy(self, rho_f, g=9.81):
        V_p = self.mass / self.density
        ndim = len(self.velocity)
        Fb = np.zeros(ndim)
        Fb[-1] = rho_f * V_p * g
        return Fb

    def gravity_force(self, g=9.81):
        ndim = len(self.velocity)
        Fg = np.zeros(ndim)
        Fg[-1] = -self.mass * g
        return Fg
    
    def calc_total_force(self, rho_fluid, mu_fluid, vel_fluid, g=9.81):
        Fd = self.calc_drag(rho_fluid, mu_fluid, vel_fluid)
        Fb = self.calc_buoyancy(rho_fluid, g)
        Fg = self.gravity_force(g)
        return Fd + Fb + Fg

    def update_velocity(self, total_force, dt):
        acceleration = total_force / self.mass
        self.velocity = self.velocity + acceleration * dt

    def update_position(self, dt):
        self.position = self.position + self.velocity * dt

    def calc_Re_p(self, rho_fluid, vel_fluid, mu_fluid):
        vel_fluid = np.atleast_1d(np.array(vel_fluid, dtype=float))
        vel_rel = vel_fluid - self.velocity
        vel_rel_mag = np.linalg.norm(vel_rel)
        Re_p = rho_fluid * vel_rel_mag * self.diameter / mu_fluid
        return Re_p

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

# Create particle
particle = solid_spherical_particle(
    diameter=0.5e-3,
    density=2560,
    initial_position=0,
    initial_velocity=0
)

# Fluid properties
rho_fluid = 1000      # kg/m3 (water)
mu_fluid = 0.0008891  # Pa·s
vel_fluid = 0         # m/s
g = 9.81
dt = 1e-6            # time step

# Convergence criteria
tolerance = 1e-8
max_iterations = 1000000

prev_velocity = 0
for i in range(max_iterations):
    F_total = particle.calc_total_force(rho_fluid, mu_fluid, vel_fluid, g)
    particle.update_velocity(F_total, dt)
    particle.update_position(dt)
    
    # Check velocity change
    vel_diff = np.linalg.norm(particle.get_velocity() - prev_velocity)
    if vel_diff < tolerance:
        print(f"Terminal velocity: {np.linalg.norm(particle.get_velocity()):.6f} m/s")
        break
    
    prev_velocity = particle.get_velocity().copy()