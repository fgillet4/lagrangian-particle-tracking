import numpy as np
import sys
sys.path.append('..')
from solid_spherical_particle import solid_spherical_particle
from particle_system import ParticleSystem
from animator import ParticleAnimator3D, create_snapshot_3d


domain_bounds = [
    (-0.05, 0.05),
    (-0.05, 0.05),
    (-0.1, 0.0)
]

fluid_properties = {
    'density': 1000,
    'viscosity': 0.0008891,
    'velocity': np.array([0.0, 0.0, 0.0]),
    'gravity': 9.81
}

system = ParticleSystem(
    domain_bounds=domain_bounds,
    fluid_properties=fluid_properties,
    dt=1e-5
)

n_particles = system.spawn_particles_random(
    n_particles=15,
    diameter_range=(0.5e-3, 1.5e-3),
    density_range=(2500, 2600),
    velocity_range=(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
)

print(f"Successfully spawned {n_particles} particles")

create_snapshot_3d(system, filename='initial_state_3d.png', 
                   elevation=30, azimuth=45, show_velocity=False)

animator = ParticleAnimator3D(system, figsize=(12, 10))

animator.simulate_and_record(n_steps=5000, record_interval=50)

print(f"Simulation complete. Creating animation...")

animator.create_animation(
    filename='particle_animation_3d.gif',
    fps=30,
    show_velocity=True,
    elevation=30,
    azimuth=45
)

print("Animation saved as 'particle_animation_3d.gif'")
