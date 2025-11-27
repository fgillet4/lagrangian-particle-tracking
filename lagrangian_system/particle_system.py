import numpy as np
import math
from typing import List, Tuple, Optional
import sys
sys.path.append('..')
from solid_spherical_particle import solid_spherical_particle


class ParticleSystem:
    def __init__(self, domain_bounds, fluid_properties, dt=1e-6):
        self.particles: List[solid_spherical_particle] = []
        self.domain_bounds = domain_bounds
        self.fluid_properties = fluid_properties
        self.dt = dt
        self.time = 0.0
        self.collision_restitution = 0.8
        
    def add_particle(self, diameter, density, initial_position, initial_velocity):
        if self._is_valid_spawn_location(initial_position, diameter):
            particle = solid_spherical_particle(
                diameter=diameter,
                density=density,
                initial_position=np.array(initial_position, dtype=float),
                initial_velocity=np.array(initial_velocity, dtype=float)
            )
            self.particles.append(particle)
            return True
        return False
    
    def spawn_particles_random(self, n_particles, diameter_range, density_range, 
                               velocity_range=(0, 0), max_attempts=1000):
        spawned = 0
        attempts = 0
        
        while spawned < n_particles and attempts < max_attempts * n_particles:
            diameter = np.random.uniform(*diameter_range)
            density = np.random.uniform(*density_range)
            
            position = self._generate_random_position(diameter)
            
            if isinstance(velocity_range[0], (list, tuple, np.ndarray)):
                velocity = np.array([
                    np.random.uniform(velocity_range[0][i], velocity_range[1][i])
                    for i in range(len(velocity_range[0]))
                ])
            else:
                velocity = np.random.uniform(velocity_range[0], velocity_range[1])
            
            if self.add_particle(diameter, density, position, velocity):
                spawned += 1
            
            attempts += 1
        
        return spawned
    
    def _generate_random_position(self, diameter):
        ndim = len(self.domain_bounds)
        position = np.zeros(ndim)
        
        for i in range(ndim):
            min_bound = self.domain_bounds[i][0] + diameter/2
            max_bound = self.domain_bounds[i][1] - diameter/2
            position[i] = np.random.uniform(min_bound, max_bound)
        
        return position
    
    def _is_valid_spawn_location(self, position, diameter):
        position = np.array(position)
        
        for i, bounds in enumerate(self.domain_bounds):
            if position[i] - diameter/2 < bounds[0] or position[i] + diameter/2 > bounds[1]:
                return False
        
        for particle in self.particles:
            distance = np.linalg.norm(position - particle.position)
            min_distance = (diameter + particle.diameter) / 2
            
            if distance < min_distance:
                return False
        
        return True
    
    def detect_particle_collisions(self):
        collisions = []
        n = len(self.particles)
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                distance = np.linalg.norm(p1.position - p2.position)
                min_distance = (p1.diameter + p2.diameter) / 2
                
                if distance < min_distance:
                    collisions.append((i, j))
        
        return collisions
    
    def resolve_particle_collision(self, i, j):
        p1 = self.particles[i]
        p2 = self.particles[j]
        
        normal = (p2.position - p1.position)
        distance = np.linalg.norm(normal)
        
        if distance == 0:
            normal = np.random.randn(len(p1.position))
            distance = np.linalg.norm(normal)
        
        normal = normal / distance
        
        overlap = (p1.diameter + p2.diameter) / 2 - distance
        p1.position -= normal * overlap / 2
        p2.position += normal * overlap / 2
        
        v_rel = p1.velocity - p2.velocity
        v_normal = np.dot(v_rel, normal)
        
        if v_normal > 0:
            return
        
        impulse = -(1 + self.collision_restitution) * v_normal
        impulse /= (1/p1.mass + 1/p2.mass)
        
        p1.velocity += impulse * normal / p1.mass
        p2.velocity -= impulse * normal / p2.mass
    
    def check_wall_collisions(self):
        for particle in self.particles:
            for i, bounds in enumerate(self.domain_bounds):
                if particle.position[i] - particle.diameter/2 < bounds[0]:
                    particle.position[i] = bounds[0] + particle.diameter/2
                    particle.velocity[i] = -particle.velocity[i] * self.collision_restitution
                
                elif particle.position[i] + particle.diameter/2 > bounds[1]:
                    particle.position[i] = bounds[1] - particle.diameter/2
                    particle.velocity[i] = -particle.velocity[i] * self.collision_restitution
    
    def step(self):
        rho_fluid = self.fluid_properties['density']
        mu_fluid = self.fluid_properties['viscosity']
        vel_fluid = self.fluid_properties.get('velocity', 0)
        g = self.fluid_properties.get('gravity', 9.81)
        
        for particle in self.particles:
            F_total = particle.calc_total_force(rho_fluid, mu_fluid, vel_fluid, g)
            particle.update_velocity(F_total, self.dt)
            particle.update_position(self.dt)
        
        collisions = self.detect_particle_collisions()
        for i, j in collisions:
            self.resolve_particle_collision(i, j)
        
        self.check_wall_collisions()
        
        self.time += self.dt
    
    def simulate(self, n_steps):
        for _ in range(n_steps):
            self.step()
    
    def get_positions(self):
        return np.array([p.position for p in self.particles])
    
    def get_velocities(self):
        return np.array([p.velocity for p in self.particles])
    
    def get_diameters(self):
        return np.array([p.diameter for p in self.particles])
