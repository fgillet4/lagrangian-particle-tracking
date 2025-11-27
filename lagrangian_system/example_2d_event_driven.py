import numpy as np
import sys
sys.path.append('..')
from solid_spherical_particle import solid_spherical_particle
from event_driven_system import EventDrivenParticleSystem
from animator import ParticleAnimator2D, create_snapshot_2d


domain_bounds = [
    (-0.05, 0.05),
    (0.0, 0.1)
]

fluid_properties = {
    'density': 1000,
    'viscosity': 0.0008891,
    'velocity': np.array([0.0, 0.0]),
    'gravity': 9.81
}

system = EventDrivenParticleSystem(
    domain_bounds=domain_bounds,
    fluid_properties=fluid_properties,
    dt=1e-5
)

n_particles = system.spawn_particles_random(
    n_particles=20,
    diameter_range=(1.0e-3, 1.0e-3),
    density_range=(2560, 2560),
    velocity_range=(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
)

print(f"Successfully spawned {n_particles} particles")
print(f"Initializing event heap...")

system.initialize_events()
print(f"Event heap initialized with {len(system.event_heap)} events")

create_snapshot_2d(system, filename='initial_state_2d_event.png', show_velocity=False)

animator = ParticleAnimator2D(system, figsize=(10, 10))

animator.record_state()

print("Starting simulation...")

for i in range(2000):
    target_time = system.time + system.dt
    system.simulate_step(target_time)
    
    if i < 500:
        if i % 5 == 0:
            animator.record_state()
    elif i < 1000:
        if i % 10 == 0:
            animator.record_state()
    else:
        if i % 50 == 0:
            animator.record_state()
    
    if i % 500 == 0:
        print(f"Step {i}, Time: {system.time:.6f}s, Heap size: {len(system.event_heap)}")

for i in range(18000):
    target_time = system.time + system.dt
    system.simulate_step(target_time)
    
    if i % 50 == 0:
        animator.record_state()
    
    if i % 5000 == 0:
        print(f"Step {2000+i}, Time: {system.time:.6f}s, Heap size: {len(system.event_heap)}")

print(f"Simulation complete. Total time: {system.time:.6f}s")
print(f"Creating animation...")

animator.create_animation(
    filename='particle_animation_2d_event.gif',
    fps=30,
    show_velocity=True,
    show_trails=True,
    trail_length=30
)

print("Animation saved as 'particle_animation_2d_event.gif'")
print(f"Total collision events processed: {sum(system.collision_counts)}")
