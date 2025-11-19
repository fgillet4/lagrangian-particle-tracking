import numpy as np
import matplotlib.pyplot as plt
import math

# Create a test particle (we just need the class methods, values don't matter much)
class solid_spherical_particle:
    # Class-level lookup table
    Re_drag_crisis = np.array([
        10472.3881, 18778.4117, 30977.1558, 47010.3815, 67015.178,
        108267.5342, 160913.6439, 239159.4209, 313643.6117, 355452.945,
        370593.1097, 386378.1546, 402835.5478, 598718.1323, 1008469.3544,
        1663586.5015, 2802113.2392, 3831332.6321, 5576823.1232
    ])
    
    Cd_drag_crisis = np.array([
        0.3996, 0.4471, 0.491, 0.5097, 0.5493, 0.5703, 0.5597, 0.5391,
        0.4819, 0.3639, 0.1999, 0.1184, 0.0829, 0.0814, 0.1, 0.13,
        0.1627, 0.1855, 0.2037
    ])
    
    @staticmethod
    def get_Cd(Re_p):
        """Calculate drag coefficient for a given Reynolds number"""
        if Re_p < 0.01:
            # Stokes: Cd = 24/Re, but handle the drag differently
            Cd = 24 / max(Re_p, 1e-10)  # Avoid division by zero
        elif Re_p < 1:
            Cd = 24 / Re_p
        elif Re_p < 1000:
            Cd = (24 / Re_p) * (1 + 0.15 * Re_p**0.687)
        elif Re_p < 10472.3881:
            Cd = 0.44
        elif Re_p <= 5576823.1232:
            Cd = np.interp(np.log10(Re_p), 
                          np.log10(solid_spherical_particle.Re_drag_crisis), 
                          solid_spherical_particle.Cd_drag_crisis)
        else:
            Cd = 0.19
        return Cd

# Generate Reynolds numbers across the full range
Re_range = np.logspace(-2, 7, 1000)  # 10^-2 to 10^7

# Calculate Cd for each Re
Cd_values = [solid_spherical_particle.get_Cd(Re) for Re in Re_range]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the continuous correlation
ax.loglog(Re_range, Cd_values, 'b-', linewidth=2.5, label='Piecewise Models')

# Overlay the digitized drag crisis points
ax.loglog(solid_spherical_particle.Re_drag_crisis, 
          solid_spherical_particle.Cd_drag_crisis, 
          'ro', markersize=8, label='Digitized Drag Crisis Data', zorder=5)

# Add regime labels
ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=1e5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Add text labels for regimes
ax.text(0.1, 100, 'Stokes\nRegime', fontsize=10, ha='center', alpha=0.7)
ax.text(10, 20, 'Transition\nRegion', fontsize=10, ha='center', alpha=0.7)
ax.text(1e4, 2, 'Newton\nRegime', fontsize=10, ha='center', alpha=0.7)
ax.text(3e5, 0.5, 'Drag\nCrisis', fontsize=10, ha='center', alpha=0.7, color='red')

# Formatting
ax.set_xlabel('Particle Reynolds Number (Re)', fontsize=14, fontweight='bold')
ax.set_ylabel('Drag Coefficient (Cd)', fontsize=14, fontweight='bold')
ax.set_title('Drag Coefficient vs Reynolds Number\nSpherical Particle Implementation', 
             fontsize=16, fontweight='bold')
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(fontsize=12, loc='upper right')
ax.set_xlim(1e-2, 1e7)
ax.set_ylim(1e-2, 1e3)

plt.tight_layout()
plt.savefig('drag_coefficient_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some test values
print("\n" + "="*60)
print("VALIDATION CHECK - Sample Cd values:")
print("="*60)
test_Re = [0.001, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 3e5, 1e6, 1e7]
for Re in test_Re:
    Cd = solid_spherical_particle.get_Cd(Re)
    print(f"Re = {Re:>10.1e}  →  Cd = {Cd:.4f}")