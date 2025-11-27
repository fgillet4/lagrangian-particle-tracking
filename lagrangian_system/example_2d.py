import numpy as np
import sys
sys.path.append('..')
from solid_spherical_particle import solid_spherical_particle
from particle_system import ParticleSystem
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

system = ParticleSystem(
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

create_snapshot_2d(system, filename='initial_state_2d.png', show_velocity=False)

animator = ParticleAnimator2D(system, figsize=(10, 10))

animator.record_state()

for i in range(2000):
    system.step()
    if i < 500:
        if i % 5 == 0:
            animator.record_state()
    elif i < 1000:
        if i % 10 == 0:
            animator.record_state()
    else:
        if i % 50 == 0:
            animator.record_state()

animator.simulate_and_record(n_steps=18000, record_interval=50)

print(f"Simulation complete. Creating animation...")

animator.create_animation(
    filename='particle_animation_2d.gif',
    fps=30,
    show_velocity=True,
    show_trails=True,
    trail_length=30
)

print("Animation saved as 'particle_animation_2d.gif'")
