import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
import sys
sys.path.append('..')
from particle_system import ParticleSystem


class ParticleAnimator2D:
    def __init__(self, particle_system: ParticleSystem, figsize=(10, 10)):
        self.system = particle_system
        self.figsize = figsize
        self.history = []
        
    def record_state(self):
        state = {
            'positions': self.system.get_positions().copy(),
            'velocities': self.system.get_velocities().copy(),
            'diameters': self.system.get_diameters().copy(),
            'time': self.system.time
        }
        self.history.append(state)
    
    def simulate_and_record(self, n_steps, record_interval=1):
        for i in range(n_steps):
            if i % record_interval == 0:
                self.record_state()
            self.system.step()
        self.record_state()
    
    def create_animation(self, filename='particle_animation_2d.gif', fps=30, 
                        show_velocity=False, show_trails=False, trail_length=50):
        if not self.history:
            raise ValueError("No recorded states. Run simulate_and_record first.")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_bounds = self.system.domain_bounds[0]
        y_bounds = self.system.domain_bounds[1]
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        
        particles = []
        velocity_arrows = []
        trails = [[] for _ in range(len(self.system.particles))]
        
        def init():
            return particles + velocity_arrows
        
        def update(frame):
            nonlocal particles, velocity_arrows
            
            for p in particles:
                p.remove()
            for arrow in velocity_arrows:
                arrow.remove()
            
            particles.clear()
            velocity_arrows.clear()
            
            state = self.history[frame]
            positions = state['positions']
            velocities = state['velocities']
            diameters = state['diameters']
            
            for i, (pos, diam) in enumerate(zip(positions, diameters)):
                circle = Circle((pos[0], pos[1]), diam/2, 
                              color='blue', alpha=0.6, zorder=2)
                ax.add_patch(circle)
                particles.append(circle)
                
                if show_trails:
                    trails[i].append(pos.copy())
                    if len(trails[i]) > trail_length:
                        trails[i].pop(0)
                    if len(trails[i]) > 1:
                        trail_array = np.array(trails[i])
                        line, = ax.plot(trail_array[:, 0], trail_array[:, 1], 
                                      'b-', alpha=0.3, linewidth=1, zorder=1)
                        particles.append(line)
                
                if show_velocity:
                    vel = velocities[i]
                    vel_mag = np.linalg.norm(vel)
                    if vel_mag > 1e-10:
                        scale = diam * 2
                        arrow = ax.arrow(pos[0], pos[1], 
                                       vel[0]/vel_mag * scale, vel[1]/vel_mag * scale,
                                       head_width=diam*0.3, head_length=diam*0.5,
                                       fc='red', ec='red', alpha=0.7, zorder=3)
                        velocity_arrows.append(arrow)
            
            ax.set_title(f'Time: {state["time"]:.6f} s | Particles: {len(positions)}')
            
            return particles + velocity_arrows
        
        anim = FuncAnimation(fig, update, init_func=init, 
                           frames=len(self.history), interval=1000/fps, 
                           blit=False, repeat=True)
        
        if filename.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        elif filename.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps)
        else:
            raise ValueError("Filename must end with .gif or .mp4")
        
        anim.save(filename, writer=writer)
        plt.close()
        
        return anim


class ParticleAnimator3D:
    def __init__(self, particle_system: ParticleSystem, figsize=(12, 10)):
        self.system = particle_system
        self.figsize = figsize
        self.history = []
        
    def record_state(self):
        state = {
            'positions': self.system.get_positions().copy(),
            'velocities': self.system.get_velocities().copy(),
            'diameters': self.system.get_diameters().copy(),
            'time': self.system.time
        }
        self.history.append(state)
    
    def simulate_and_record(self, n_steps, record_interval=1):
        for i in range(n_steps):
            if i % record_interval == 0:
                self.record_state()
            self.system.step()
        self.record_state()
    
    def create_animation(self, filename='particle_animation_3d.gif', fps=30, 
                        show_velocity=False, elevation=30, azimuth=45):
        if not self.history:
            raise ValueError("No recorded states. Run simulate_and_record first.")
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        x_bounds = self.system.domain_bounds[0]
        y_bounds = self.system.domain_bounds[1]
        z_bounds = self.system.domain_bounds[2]
        
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_zlim(z_bounds)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        def update(frame):
            ax.clear()
            ax.set_xlim(x_bounds)
            ax.set_ylim(y_bounds)
            ax.set_zlim(z_bounds)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.view_init(elev=elevation, azim=azimuth)
            
            state = self.history[frame]
            positions = state['positions']
            velocities = state['velocities']
            diameters = state['diameters']
            
            for i, (pos, diam) in enumerate(zip(positions, diameters)):
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = pos[0] + (diam/2) * np.outer(np.cos(u), np.sin(v))
                y = pos[1] + (diam/2) * np.outer(np.sin(u), np.sin(v))
                z = pos[2] + (diam/2) * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_surface(x, y, z, color='blue', alpha=0.6, shade=True)
                
                if show_velocity:
                    vel = velocities[i]
                    vel_mag = np.linalg.norm(vel)
                    if vel_mag > 1e-10:
                        scale = diam * 2
                        ax.quiver(pos[0], pos[1], pos[2],
                                vel[0], vel[1], vel[2],
                                length=scale/vel_mag, color='red', 
                                arrow_length_ratio=0.3, linewidth=2)
            
            ax.set_title(f'Time: {state["time"]:.6f} s | Particles: {len(positions)}')
            
            return ax,
        
        anim = FuncAnimation(fig, update, frames=len(self.history), 
                           interval=1000/fps, blit=False, repeat=True)
        
        if filename.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        elif filename.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps)
        else:
            raise ValueError("Filename must end with .gif or .mp4")
        
        anim.save(filename, writer=writer)
        plt.close()
        
        return anim


def create_snapshot_2d(particle_system: ParticleSystem, filename='snapshot_2d.png', 
                       figsize=(10, 10), show_velocity=False):
    fig, ax = plt.subplots(figsize=figsize)
    
    x_bounds = particle_system.domain_bounds[0]
    y_bounds = particle_system.domain_bounds[1]
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    
    positions = particle_system.get_positions()
    velocities = particle_system.get_velocities()
    diameters = particle_system.get_diameters()
    
    for i, (pos, diam) in enumerate(zip(positions, diameters)):
        circle = Circle((pos[0], pos[1]), diam/2, color='blue', alpha=0.6)
        ax.add_patch(circle)
        
        if show_velocity:
            vel = velocities[i]
            vel_mag = np.linalg.norm(vel)
            if vel_mag > 1e-10:
                scale = diam * 2
                ax.arrow(pos[0], pos[1], 
                       vel[0]/vel_mag * scale, vel[1]/vel_mag * scale,
                       head_width=diam*0.3, head_length=diam*0.5,
                       fc='red', ec='red', alpha=0.7)
    
    ax.set_title(f'Time: {particle_system.time:.6f} s | Particles: {len(positions)}')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def create_snapshot_3d(particle_system: ParticleSystem, filename='snapshot_3d.png',
                       figsize=(12, 10), elevation=30, azimuth=45, show_velocity=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    x_bounds = particle_system.domain_bounds[0]
    y_bounds = particle_system.domain_bounds[1]
    z_bounds = particle_system.domain_bounds[2]
    
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_zlim(z_bounds)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=elevation, azim=azimuth)
    
    positions = particle_system.get_positions()
    velocities = particle_system.get_velocities()
    diameters = particle_system.get_diameters()
    
    for i, (pos, diam) in enumerate(zip(positions, diameters)):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = pos[0] + (diam/2) * np.outer(np.cos(u), np.sin(v))
        y = pos[1] + (diam/2) * np.outer(np.sin(u), np.sin(v))
        z = pos[2] + (diam/2) * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, color='blue', alpha=0.6, shade=True)
        
        if show_velocity:
            vel = velocities[i]
            vel_mag = np.linalg.norm(vel)
            if vel_mag > 1e-10:
                scale = diam * 2
                ax.quiver(pos[0], pos[1], pos[2],
                        vel[0], vel[1], vel[2],
                        length=scale/vel_mag, color='red', 
                        arrow_length_ratio=0.3, linewidth=2)
    
    ax.set_title(f'Time: {particle_system.time:.6f} s | Particles: {len(positions)}')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
