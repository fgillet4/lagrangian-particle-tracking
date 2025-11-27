# Lagrangian Particle Tracking System

This folder contains a complete Lagrangian particle tracking system for simulating spherical particles in Eulerian fluid fields.

## Features

- **ParticleSystem**: Manages multiple particles with:
  - Collision detection (particle-particle and particle-wall)
  - Non-overlapping spawn locations
  - Physics-based motion using forces from Assignment 1
  
- **Animation**: Generate 2D and 3D animations with:
  - GIF and MP4 export
  - Velocity vectors
  - Particle trails
  - Customizable views

## Files

- `particle_system.py` - Main particle system class with collision detection
- `animator.py` - 2D and 3D animation tools
- `example_2d.py` - 2D simulation example
- `example_3d.py` - 3D simulation example

## Usage

### 2D Example
```python
python example_2d.py
```

### 3D Example
```python
python example_3d.py
```

### Custom Simulation
```python
from particle_system import ParticleSystem
from animator import ParticleAnimator2D

# Define domain and fluid
domain_bounds = [(-0.05, 0.05), (-0.1, 0.0)]
fluid_properties = {
    'density': 1000,
    'viscosity': 0.0008891,
    'velocity': [0.0, 0.0],
    'gravity': 9.81
}

# Create system
system = ParticleSystem(domain_bounds, fluid_properties, dt=1e-5)

# Spawn particles
system.spawn_particles_random(
    n_particles=20,
    diameter_range=(0.5e-3, 1.5e-3),
    density_range=(2500, 2600)
)

# Animate
animator = ParticleAnimator2D(system)
animator.simulate_and_record(n_steps=5000, record_interval=50)
animator.create_animation('output.gif', fps=30, show_velocity=True)
```

## Requirements

- numpy
- matplotlib
- solid_spherical_particle (from parent directory)
