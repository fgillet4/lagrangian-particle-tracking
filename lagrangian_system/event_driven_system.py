import numpy as np
import heapq
import math
from typing import List, Tuple, Optional
import sys
sys.path.append('..')
from solid_spherical_particle import solid_spherical_particle


class CollisionEvent:
    def __init__(self, time, particle_a_idx, particle_b_idx=None, wall_axis=None, wall_dir=None, count_a=0, count_b=0):
        self.time = time
        self.particle_a_idx = particle_a_idx
        self.particle_b_idx = particle_b_idx
        self.wall_axis = wall_axis
        self.wall_dir = wall_dir
        self.count_a = count_a
        self.count_b = count_b
    
    def __lt__(self, other):
        return self.time < other.time
    
    def is_particle_collision(self):
        return self.particle_b_idx is not None
    
    def is_wall_collision(self):
        return self.wall_axis is not None
    
    def is_valid(self, collision_counts):
        if self.is_particle_collision():
            return (collision_counts[self.particle_a_idx] == self.count_a and 
                    collision_counts[self.particle_b_idx] == self.count_b)
        else:
            return collision_counts[self.particle_a_idx] == self.count_a


class EventDrivenParticleSystem:
    def __init__(self, domain_bounds, fluid_properties, dt=1e-5):
        self.particles: List[solid_spherical_particle] = []
        self.domain_bounds = domain_bounds
        self.fluid_properties = fluid_properties
        self.dt = dt
        self.time = 0.0
        self.collision_restitution = 0.8
        
        self.event_heap = []
        self.collision_counts = []
        self.last_force_update = []
        self.forces = []
        
    def add_particle(self, diameter, density, initial_position, initial_velocity):
        if self._is_valid_spawn_location(initial_position, diameter):
            particle = solid_spherical_particle(
                diameter=diameter,
                density=density,
                initial_position=np.array(initial_position, dtype=float),
                initial_velocity=np.array(initial_velocity, dtype=float)
            )
            self.particles.append(particle)
            self.collision_counts.append(0)
            self.last_force_update.append(0.0)
            
            rho_fluid = self.fluid_properties['density']
            mu_fluid = self.fluid_properties['viscosity']
            vel_fluid = self.fluid_properties.get('velocity', 0)
            g = self.fluid_properties.get('gravity', 9.81)
            F_total = particle.calc_total_force(rho_fluid, mu_fluid, vel_fluid, g)
            self.forces.append(F_total)
            
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
    
    def predict_particle_collision(self, i, j):
        p1 = self.particles[i]
        p2 = self.particles[j]
        
        dpos = p2.position - p1.position
        dvel = p2.velocity - p1.velocity
        
        sigma = (p1.diameter + p2.diameter) / 2
        
        dv_dot_dr = np.dot(dvel, dpos)
        if dv_dot_dr >= 0:
            return None
        
        dv_dot_dv = np.dot(dvel, dvel)
        dr_dot_dr = np.dot(dpos, dpos)
        
        discriminant = dv_dot_dr**2 - dv_dot_dv * (dr_dot_dr - sigma**2)
        
        if discriminant < 0:
            return None
        
        dt = -(dv_dot_dr + np.sqrt(discriminant)) / dv_dot_dv
        
        if dt <= 0:
            return None
        
        return dt
    
    def predict_wall_collision(self, i):
        particle = self.particles[i]
        min_time = float('inf')
        wall_info = None
        
        for axis, bounds in enumerate(self.domain_bounds):
            if particle.velocity[axis] > 0:
                dt = (bounds[1] - particle.position[axis] - particle.diameter/2) / particle.velocity[axis]
                if 0 < dt < min_time:
                    min_time = dt
                    wall_info = (axis, 1)
            elif particle.velocity[axis] < 0:
                dt = (bounds[0] - particle.position[axis] + particle.diameter/2) / particle.velocity[axis]
                if 0 < dt < min_time:
                    min_time = dt
                    wall_info = (axis, -1)
        
        if wall_info is None:
            return None
        
        return min_time, wall_info[0], wall_info[1]
    
    def initialize_events(self):
        self.event_heap.clear()
        
        n = len(self.particles)
        for i in range(n):
            for j in range(i + 1, n):
                dt = self.predict_particle_collision(i, j)
                if dt is not None:
                    event = CollisionEvent(
                        self.time + dt, i, j, 
                        count_a=self.collision_counts[i],
                        count_b=self.collision_counts[j]
                    )
                    heapq.heappush(self.event_heap, event)
            
            wall_result = self.predict_wall_collision(i)
            if wall_result is not None:
                dt, axis, direction = wall_result
                event = CollisionEvent(
                    self.time + dt, i, 
                    wall_axis=axis, wall_dir=direction,
                    count_a=self.collision_counts[i]
                )
                heapq.heappush(self.event_heap, event)
    
    def update_particle_to_time(self, i, target_time):
        particle = self.particles[i]
        dt = target_time - self.time
        
        if dt <= 0:
            return
        
        particle.position = particle.position + particle.velocity * dt
    
    def update_all_particles_to_time(self, target_time):
        for i in range(len(self.particles)):
            self.update_particle_to_time(i, target_time)
        self.time = target_time
    
    def apply_forces_with_drag(self, duration):
        rho_fluid = self.fluid_properties['density']
        mu_fluid = self.fluid_properties['viscosity']
        vel_fluid = self.fluid_properties.get('velocity', 0)
        g = self.fluid_properties.get('gravity', 9.81)
        
        for i, particle in enumerate(self.particles):
            F_total = particle.calc_total_force(rho_fluid, mu_fluid, vel_fluid, g)
            acceleration = F_total / particle.mass
            particle.velocity = particle.velocity + acceleration * duration
    
    def resolve_particle_collision(self, i, j):
        p1 = self.particles[i]
        p2 = self.particles[j]
        
        normal = (p2.position - p1.position)
        distance = np.linalg.norm(normal)
        
        if distance == 0:
            normal = np.random.randn(len(p1.position))
            distance = np.linalg.norm(normal)
        
        normal = normal / distance
        
        v_rel = p1.velocity - p2.velocity
        v_normal = np.dot(v_rel, normal)
        
        if v_normal > 0:
            return
        
        impulse = -(1 + self.collision_restitution) * v_normal
        impulse /= (1/p1.mass + 1/p2.mass)
        
        p1.velocity += impulse * normal / p1.mass
        p2.velocity -= impulse * normal / p2.mass
        
        self.collision_counts[i] += 1
        self.collision_counts[j] += 1
    
    def resolve_wall_collision(self, i, axis):
        particle = self.particles[i]
        particle.velocity[axis] = -particle.velocity[axis] * self.collision_restitution
        self.collision_counts[i] += 1
    
    def predict_and_add_events(self, particle_idx):
        n = len(self.particles)
        
        for j in range(n):
            if j == particle_idx:
                continue
            
            i = min(particle_idx, j)
            k = max(particle_idx, j)
            
            dt = self.predict_particle_collision(i, k)
            if dt is not None:
                event = CollisionEvent(
                    self.time + dt, i, k,
                    count_a=self.collision_counts[i],
                    count_b=self.collision_counts[k]
                )
                heapq.heappush(self.event_heap, event)
        
        wall_result = self.predict_wall_collision(particle_idx)
        if wall_result is not None:
            dt, axis, direction = wall_result
            event = CollisionEvent(
                self.time + dt, particle_idx,
                wall_axis=axis, wall_dir=direction,
                count_a=self.collision_counts[particle_idx]
            )
            heapq.heappush(self.event_heap, event)
    
    def simulate_step(self, target_time):
        while self.event_heap and self.event_heap[0].time <= target_time:
            event = heapq.heappop(self.event_heap)
            
            if not event.is_valid(self.collision_counts):
                continue
            
            self.update_all_particles_to_time(event.time)
            
            if event.is_particle_collision():
                self.resolve_particle_collision(event.particle_a_idx, event.particle_b_idx)
                self.predict_and_add_events(event.particle_a_idx)
                self.predict_and_add_events(event.particle_b_idx)
            else:
                self.resolve_wall_collision(event.particle_a_idx, event.wall_axis)
                self.predict_and_add_events(event.particle_a_idx)
        
        self.update_all_particles_to_time(target_time)
        
        time_since_last = target_time - (target_time - self.dt)
        self.apply_forces_with_drag(self.dt)
    
    def simulate(self, n_steps):
        self.initialize_events()
        for _ in range(n_steps):
            target_time = self.time + self.dt
            self.simulate_step(target_time)
    
    def get_positions(self):
        return np.array([p.position for p in self.particles])
    
    def get_velocities(self):
        return np.array([p.velocity for p in self.particles])
    
    def get_diameters(self):
        return np.array([p.diameter for p in self.particles])
