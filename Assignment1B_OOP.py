#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object-oriented refactor of bubble column simulation

__authors__ = "Ananda S. Kannan and Niklas Hidman"
__institution__ = "Department of Mechanics and Maritime Sciences, Chalmers University of Technology, Sweden"
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import numpy as np
import os

np.random.seed(12345)

class Bubble:
    
    def __init__(self, bubble_id, diameter, x_pos, injection_time):
        self.id = bubble_id
        self.D = diameter
        self.mass = None
        
        self.x = x_pos
        self.y = 0.0
        self.u = 0.0
        self.v = 0.0
        
        self.Re = 0.0
        self.Eo = 0.0
        self.We = 0.0
        self.Mo = 0.0
        
        self.E = 1.0
        self.d_horizontal = diameter
        self.d_vertical = diameter
        self.shape_regime = "spherical"
        
        self.Fb = 0.0
        self.Fdy = 0.0
        self.Fdx = 0.0
        self.Fp = 0.0
        self.Fg = 0.0
        self.FLx = 0.0
        self.Fhist = 0.0
        self.Fhist_x = 0.0
        self.F_W = 0.0
        
        self.injection_time = injection_time
        self.deletion_time = None
        self.alive = True
        
        self.x_history = [x_pos]
        self.y_history = [0.0]
        self.u_history = [0.0]
        self.v_history = [0.0]
        self.E_history = [1.0]
        self.shape_regime_history = ["spherical"]
        
    def set_mass(self, rhoB):
        self.mass = rhoB * 4.0/3 * np.pi * (self.D/2.0)**3
        
    def update_position(self, dt):
        self.x += dt * self.u
        self.y += dt * self.v
        
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.u_history.append(self.u)
        self.v_history.append(self.v)
        self.E_history.append(self.E)
        self.shape_regime_history.append(self.shape_regime)
        
    def update_shape(self, Re, Eo, We, Mo, E, shape_regime):
        self.Re = Re
        self.Eo = Eo
        self.We = We
        self.Mo = Mo
        self.E = E
        self.d_horizontal = self.D / (E**(1/3))
        self.d_vertical = self.D * E**(2/3)
        self.shape_regime = shape_regime
    
    def update_forces(self, Fb, Fdy, Fdx, Fp, Fg, FLx, Fhist, Fhist_x, F_W):
        self.Fb = Fb
        self.Fdy = Fdy
        self.Fdx = Fdx
        self.Fp = Fp
        self.Fg = Fg
        self.FLx = FLx
        self.Fhist = Fhist
        self.Fhist_x = Fhist_x
        self.F_W = F_W
        
    def mark_deleted(self, deletion_time):
        self.alive = False
        self.deletion_time = deletion_time

class BubbleColumn:
    
    def __init__(self, b, L, rhoL, mul, sig, rhoB, g, py, mean_dia, std_dia):
        self.b = b
        self.L = L
        self.rhoL = rhoL
        self.mul = mul
        self.sig = sig
        self.rhoB = rhoB
        self.g = g
        self.py = py
        self.mean_dia = mean_dia
        self.std_dia = std_dia
        
        self.bubbles = []
        self.bubble_counter = 0
        
    def sample_bubble_diameter(self):
        return np.random.normal(self.mean_dia, self.std_dia)
    
    def inject_bubble(self, x_pos, t):
        D = self.sample_bubble_diameter()
        bubble = Bubble(self.bubble_counter, D, x_pos, t)
        bubble.set_mass(self.rhoB)
        self.bubbles.append(bubble)
        self.bubble_counter += 1
        return bubble
    
    def fluid_velocity_and_gradient(self, x):
        Vy = self.py * x / (2*self.mul) * (2*self.b - x)
        dVdx = self.py / self.mul * (self.b - x)
        return Vy, dVdx
    
    def calculate_dimensionless_numbers(self, bubble, Vrel):
        Re = self.rhoL * np.sqrt(Vrel**2 + bubble.u**2) * bubble.D / self.mul
        Eo = (self.rhoL - self.rhoB) * self.g * bubble.D**2 / self.sig
        Mo = self.g * self.mul**4 * (self.rhoL - self.rhoB) / (self.rhoL**2 * self.sig**3)
        We = self.rhoL * (np.sqrt(Vrel**2 + bubble.u**2))**2 * bubble.D / self.sig
        return Re, Eo, Mo, We
    
    def calculate_aspect_ratio(self, Eo):
        E = 1.0 / (1.0 + 0.163 * Eo**0.757)
        return E
    
    def classify_shape_regime(self, Eo, We):
        if Eo < 0.1:
            return "spherical"
        elif Eo < 4.0:
            return "ellipsoidal"
        elif Eo < 40.0:
            if We < 3.0:
                return "ellipsoidal"
            else:
                return "wobbling"
        else:
            return "spherical-cap"
    
    def calculate_drag_coefficient(self, Re, Eo, shape_regime, contaminated=False):
        if contaminated:
            beta = 1
            Cd = beta * max(min(16/Re*(1+0.15*Re**0.687), 48/Re), 8/3*Eo/(Eo+4))
        else:
            if shape_regime == "spherical":
                mu_ratio = 0.0
                Cd_HR = (24/Re) * (2/3 + mu_ratio) / (1 + mu_ratio)
                Cd = max(Cd_HR, 8/3*Eo/(Eo+4))
            elif shape_regime == "ellipsoidal":
                Cd = (16/Re) * (1 + 0.15*Re**0.687)
            elif shape_regime == "wobbling":
                Cd = max(8/3*Eo/(Eo+4), 0.44)
            else:
                Cd = 8/3*Eo/(Eo+4)
        return Cd
    
    def calculate_lift_coefficient(self, Re, Eo):
        D_eq = np.sqrt(Eo * self.sig / (self.rhoL * self.g))
        D_h = D_eq * (1 + 0.163 * Eo**0.757)**(1/3)
        Eoh = self.rhoL * self.g * D_h**2 / self.sig
        feo = 0.00105*Eoh**3 - 0.0159*Eoh**2 - 0.0204*Eoh + 0.474
        
        if Eoh < 4.0:
            Cl = np.min([0.288*np.tanh(0.121*Re), feo])
        elif Eoh < 10.7:
            Cl = feo
        else:
            Eoh = 10.7
            Cl = 0.00105*Eoh**3 - 0.0159*Eoh**2 - 0.0204*Eoh + 0.474
        
        return Cl
    
    def calculate_wall_force_factor(self, x, D, min_dist):
        x_left = max(x, min_dist)
        x_right = max(2*self.b - x, min_dist)
        
        if x_left < min_dist or x_right < min_dist:
            f_w_left = -D / (min_dist**2) if x < min_dist else 0
            f_w_right = D / (min_dist**2) if (2*self.b - x) < min_dist else 0
        else:
            f_w_left = -D / (x_left**2)
            f_w_right = D / (x_right**2)
        
        return f_w_left + f_w_right
    
    def update_bubble(self, bubble, t, dt, C_W=0.00025):
        Vy, dVdx = self.fluid_velocity_and_gradient(bubble.x)
        
        if bubble.D < 1.3e-3:
            Vrel = np.sqrt(2*self.sig/(self.rhoL*bubble.D) + 
                          (self.rhoL - self.rhoB)*self.g*bubble.D/(2*self.rhoL))
        else:
            Vrel = bubble.v - Vy
        
        Re, Eo, Mo, We = self.calculate_dimensionless_numbers(bubble, Vrel)
        E = self.calculate_aspect_ratio(Eo)
        shape_regime = self.classify_shape_regime(Eo, We)
        
        bubble.update_shape(Re, Eo, We, Mo, E, shape_regime)
        
        Cd = self.calculate_drag_coefficient(Re, Eo, shape_regime, contaminated=False)
        Cl = self.calculate_lift_coefficient(Re, Eo)
        
        Fb = self.rhoL * (4/3*np.pi*(bubble.D/2)**3) * self.g
        Fdy = 0.5 * self.rhoL * Cd * np.pi*bubble.D**2/4 * np.sqrt(Vrel**2 + bubble.u**2) * Vrel
        Fp = bubble.mass * self.rhoL / self.rhoB * (-self.g)
        Fg = bubble.mass * self.g
        
        t_since_injection = t - bubble.injection_time
        if t_since_injection > 0.0:
            mhist = np.sqrt(self.rhoL*self.mul*np.pi) * bubble.mass / (self.rhoB*bubble.D)
            Fhist = mhist * Vrel / np.sqrt(0.5*t_since_injection)
        else:
            Fhist = 0
        
        omega = np.array([0, 0, dVdx])
        Vrel_vec = np.array([0, Vrel, 0])
        cross_prod = np.cross(Vrel_vec, omega)
        
        FLx = -Cl * self.rhoL * (np.pi*bubble.D**3)/6 * (bubble.v - Vy) * cross_prod[0]
        Fdx = 0.5 * self.rhoL * Cd * (np.pi*bubble.D**2/4) * np.sqrt(Vrel**2 + bubble.u**2) * bubble.u
        
        if t_since_injection > 0.0:
            Fhist_x = mhist * (-bubble.u) / np.sqrt(0.5*t_since_injection)
        else:
            Fhist_x = 0
        
        min_dist = 5 * bubble.D
        f_w = self.calculate_wall_force_factor(bubble.x, bubble.D, min_dist)
        F_W = C_W * f_w * 0.5 * self.rhoL * (np.pi*bubble.D**2/4) * np.sqrt(Vrel**2 + bubble.u**2)**2
        
        bubble.update_forces(Fb, Fdy, Fdx, Fp, Fg, FLx, Fhist, Fhist_x, F_W)
        
        FtotY = bubble.Fb + bubble.Fp - bubble.Fdy - bubble.Fhist - bubble.Fg
        FtotX = bubble.FLx - bubble.Fdx - bubble.Fhist_x + bubble.F_W
        
        totMass = bubble.mass + (0.5*self.rhoL*(4/3*np.pi*(bubble.D/2)**3))
        
        if not np.isfinite(FtotX):
            FtotX = 0.0
        
        bubble.v += dt * (FtotY / totMass)
        bubble.u += dt * (FtotX / totMass)
        
        bubble.update_position(dt)
        
        if bubble.x < bubble.D/2.0 and bubble.u < 0.0:
            bubble.u = 0.0
            bubble.x -= dt * bubble.u
        elif bubble.x > (2*self.b - bubble.D/2.0) and bubble.u > 0.0:
            bubble.u = 0.0
            bubble.x -= dt * bubble.u
        
        if bubble.y > self.L:
            bubble.mark_deleted(t)
            return False
        
        return True
    
    def remove_dead_bubbles(self):
        self.bubbles = [b for b in self.bubbles if b.alive]

def run_simulation(tEnd=30.0, dt=1e-2, n_nozzles=6, massFlowRateTot=2.4e-6, C_W=0.00025, ppPath=None):
    
    column = BubbleColumn(
        b=0.025, L=3.0, rhoL=1000.0, mul=0.001, sig=0.073,
        rhoB=1.2, g=9.82, py=1, mean_dia=0.0058, std_dia=0.0015
    )
    
    n_timeSteps = int(np.ceil(tEnd/dt))
    times = np.linspace(0, tEnd, n_timeSteps)
    
    massMeanBubble = column.rhoB * 4/3 * np.pi * (column.mean_dia/2.0)**3
    massFlowRateNozzle = massFlowRateTot / n_nozzles
    bubbleInjectionFrequency = massFlowRateNozzle / massMeanBubble
    
    timeSinceInjection = 1.0
    
    if ppPath is None:
        ppPath = os.path.join(os.getcwd(), "results")
    if not os.path.exists(ppPath):
        os.makedirs(ppPath)
    
    for ti, t in enumerate(times):
        
        if timeSinceInjection > 1.0/bubbleInjectionFrequency:
            timeSinceInjection = 0
            for noz in range(n_nozzles):
                injectionPos = 2.0*column.b/(n_nozzles+1) * (noz+1)
                column.inject_bubble(injectionPos, t)
        else:
            timeSinceInjection += dt
        
        for bubble in column.bubbles:
            if bubble.alive:
                column.update_bubble(bubble, t, dt, C_W=C_W)
        
        column.remove_dead_bubbles()
    
    return column, times

def plot_results(column, times):
    
    ppPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/TME160/bubbleColumn"
    
    plt.rcParams.update({'font.size': 10})
    plt.rcParams["font.family"] = "serif"
    
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Bubble Column Simulation (OOP)\nForces: Buoyancy, Drag, History, Pressure Gradient, Lift (Tomiyama), Wall Force', 
                 fontsize=12, fontweight='bold')
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3, top=0.88)
    
    VyToPlot, dVdxToPlot = column.fluid_velocity_and_gradient(np.linspace(0, 2*column.b, 100))
    
    max_dia = max(b.D for b in column.bubbles)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlabel('x-coord')
    ax1.set_ylabel('y-coord')
    for bubble in column.bubbles:
        ax1.plot(bubble.x_history, bubble.y_history, 
                color=cm.jet(bubble.D/max_dia))
    
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Velocity profile')
    ax1_twin.plot(np.linspace(0, 2*column.b, 100), VyToPlot, '--', color='red', linewidth=2)
    ax1.set_title('Bubble trajectory (colored by bubble size)')
    ax1.grid(True)
    
    binsInXdir = 10
    dy = 0.02
    n_timeSteps = len(times)
    
    def calculate_void_fraction_at_time(column, y_center, dy, n_bins, time_idx):
        voidFractionPerBin = np.zeros(n_bins)
        for iBin in range(n_bins):
            xMax = (iBin+1) * 2.0*column.b / n_bins
            xMin = iBin * 2.0*column.b / n_bins
            
            voidArea = 0
            for bubble in column.bubbles:
                if len(bubble.y_history) > time_idx:
                    y = bubble.y_history[time_idx]
                    x = bubble.x_history[time_idx]
                    if (y > y_center - dy) and (y < y_center):
                        if (x > xMin) and (x < xMax):
                            voidArea += bubble.D**2 * np.pi/4.0
            
            totArea = (xMax - xMin) * dy
            voidFractionPerBin[iBin] = voidArea / totArea
        
        return voidFractionPerBin
    
    avVoidFracArea1 = np.zeros([n_timeSteps, binsInXdir])
    avVoidFracArea2 = np.zeros([n_timeSteps, binsInXdir])
    avVoidFracArea3 = np.zeros([n_timeSteps, binsInXdir])
    
    for i in range(n_timeSteps):
        avVoidFracArea1[i,:] = calculate_void_fraction_at_time(column, 0.3, dy, binsInXdir, i)
        avVoidFracArea2[i,:] = calculate_void_fraction_at_time(column, 1.0, dy, binsInXdir, i)
        avVoidFracArea3[i,:] = calculate_void_fraction_at_time(column, 2.0, dy, binsInXdir, i)
    
    startAverTimeIndex = int(0.3 * n_timeSteps)
    timeAverageOverBins1 = np.mean(avVoidFracArea1[startAverTimeIndex:-1,:], axis=0)
    timeAverageOverBins2 = np.mean(avVoidFracArea2[startAverTimeIndex:-1,:], axis=0)
    timeAverageOverBins3 = np.mean(avVoidFracArea3[startAverTimeIndex:-1,:], axis=0)
    
    ax2 = fig.add_subplot(gs[0, 1])
    x_pos_bins = [i*2*column.b/binsInXdir + 2*column.b/(2*binsInXdir) for i in range(binsInXdir)]
    ax2.plot(x_pos_bins, timeAverageOverBins1, '-*')
    ax2.plot(x_pos_bins, timeAverageOverBins2, '-*')
    ax2.plot(x_pos_bins, timeAverageOverBins3, '-*')
    ax2.set_xlabel('x-pos')
    ax2.set_ylabel('time av void fraction')
    ax2.legend(['y = 0.3m','y = 1.0m','y = 2.0m'])
    ax2.set_title('Area avg void fraction')
    ax2.grid(True)
    
    ax3 = fig.add_subplot(gs[0, 2])
    timeIndexToPlot = int(0.5 * n_timeSteps)
    for bubble in column.bubbles:
        if len(bubble.y_history) > timeIndexToPlot:
            if bubble.y_history[timeIndexToPlot] > 0.0:
                ax3.plot(bubble.x_history[timeIndexToPlot], bubble.y_history[timeIndexToPlot], 
                        'o', color=cm.jet(bubble.D/max_dia))
    ax3.set_ylim([0, column.L])
    ax3.set_xlabel('x-pos') 
    ax3.set_ylabel('y-pos') 
    ax3.set_title('Bubble pos snapshot')
    ax3.grid(True)
    
    figName = "Bubble_column_results_OOP.png"
    plt.savefig(os.path.join(ppPath, figName), dpi=250, bbox_inches='tight')
    plt.close(fig)

class BubbleAnimator:
    
    def __init__(self, column, times, ppPath):
        self.column = column
        self.times = times
        self.ppPath = ppPath
    
    def create_snapshot(self, time_idx, save=True):
        plt.rcParams.update({'font.size': 10})
        plt.rcParams["font.family"] = "serif"
        
        fig, ax = plt.subplots(figsize=(3, 12))
        
        ax.set_xlim([0, 2*self.column.b])
        ax.set_ylim([0, self.column.L])
        ax.set_xlabel('x-position (m)')
        ax.set_ylabel('y-position (m)')
        ax.set_title(f'Bubble Column - t = {self.times[time_idx]:.2f}s')
        ax.grid(True, alpha=0.3)
        
        n_nozzles = 6
        for noz in range(n_nozzles):
            injectionPos = 2.0*self.column.b/(n_nozzles+1) * (noz+1)
            ax.plot(injectionPos, 0, 'r^', markersize=8, label='Injector' if noz==0 else '')
        
        for bubble in self.column.bubbles:
            if len(bubble.y_history) > time_idx:
                x = bubble.x_history[time_idx]
                y = bubble.y_history[time_idx]
                
                if y > 0 and y < self.column.L and x > 0 and x < 2*self.column.b:
                    circle = plt.Circle((x, y), bubble.D/2, facecolor='cyan', alpha=0.6, edgecolor='darkblue', linewidth=0.5)
                    ax.add_patch(circle)
        
        if save:
            filename = os.path.join(self.snapshot_dir, f"snapshot_{time_idx:04d}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return filename
        else:
            return fig
    
    def create_animation_mp4(self, frame_skip=10, fps=10):
        print(f"Creating MP4 animation (every {frame_skip} frames at {fps} fps)...")
        
        fig, ax = plt.subplots(figsize=(4, 10))
        ax.set_xlim([0, 2*self.column.b])
        ax.set_ylim([0, self.column.L])
        ax.set_xlabel('x-position (m)')
        ax.set_ylabel('y-position (m)')
        ax.grid(True, alpha=0.3)
        
        n_nozzles = 6
        for noz in range(n_nozzles):
            injectionPos = 2.0*self.column.b/(n_nozzles+1) * (noz+1)
            ax.plot(injectionPos, 0, 'r^', markersize=8)
        
        scatter = ax.scatter([], [], s=[], c='cyan', alpha=0.6, edgecolors='darkblue', linewidths=0.5)
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
        
        frame_indices = list(range(0, len(self.times), frame_skip))
        
        def update(frame_num):
            time_idx = frame_indices[frame_num]
            
            x_positions = []
            y_positions = []
            sizes = []
            
            for bubble in self.column.bubbles:
                if len(bubble.y_history) > time_idx:
                    x = bubble.x_history[time_idx]
                    y = bubble.y_history[time_idx]
                    
                    if y > 0 and y < self.column.L and x > 0 and x < 2*self.column.b:
                        x_positions.append(x)
                        y_positions.append(y)
                        sizes.append(200)
            
            scatter.set_offsets(np.c_[x_positions, y_positions])
            scatter.set_sizes(sizes)
            time_text.set_text(f't = {self.times[time_idx]:.2f}s')
            
            if frame_num % 10 == 0:
                print(f"  Frame {frame_num}/{len(frame_indices)}")
            
            return scatter, time_text
        
        anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000/fps, blit=True)
        
        mp4_path = os.path.join(self.ppPath, "bubble_column_animation.mp4")
        anim.save(mp4_path, writer='ffmpeg', fps=fps, dpi=150)
        plt.close(fig)
        
        print(f"MP4 saved to: {mp4_path}")
        return mp4_path
    
    def create_animation_with_deformation(self, frame_skip=10, fps=10):
        print(f"Creating deformation MP4 animation (every {frame_skip} frames at {fps} fps)...")
        
        fig, ax = plt.subplots(figsize=(4, 10))
        ax.set_xlim([0, 2*self.column.b])
        ax.set_ylim([0, self.column.L])
        ax.set_xlabel('x-position (m)')
        ax.set_ylabel('y-position (m)')
        ax.grid(True, alpha=0.3)
        
        n_nozzles = 6
        for noz in range(n_nozzles):
            injectionPos = 2.0*self.column.b/(n_nozzles+1) * (noz+1)
            ax.plot(injectionPos, 0, 'r^', markersize=8)
        
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=10)
        shape_text = ax.text(0.02, 0.93, '', transform=ax.transAxes, va='top', fontsize=8)
        
        frame_indices = list(range(0, len(self.times), frame_skip))
        scatter = ax.scatter([], [], s=[], c=[], alpha=0.6, edgecolors='darkblue', linewidths=0.5)
        
        shape_colors = {
            'spherical': 'cyan',
            'ellipsoidal': 'yellow',
            'wobbling': 'orange',
            'spherical-cap': 'red'
        }
        
        def update(frame_num):
            time_idx = frame_indices[frame_num]
            
            x_positions = []
            y_positions = []
            sizes = []
            colors = []
            shape_counts = {'spherical': 0, 'ellipsoidal': 0, 'wobbling': 0, 'spherical-cap': 0}
            
            for bubble in self.column.bubbles:
                if len(bubble.y_history) > time_idx:
                    x = bubble.x_history[time_idx]
                    y = bubble.y_history[time_idx]
                    
                    if y > 0 and y < self.column.L and x > 0 and x < 2*self.column.b:
                        E = bubble.E_history[time_idx]
                        shape_regime = bubble.shape_regime_history[time_idx]
                        
                        x_positions.append(x)
                        y_positions.append(y)
                        
                        sizes.append(250)
                        
                        colors.append(shape_colors.get(shape_regime, 'cyan'))
                        shape_counts[shape_regime] += 1
            
            scatter.set_offsets(np.c_[x_positions, y_positions] if x_positions else np.empty((0, 2)))
            scatter.set_sizes(sizes)
            scatter.set_facecolors(colors)
            
            time_text.set_text(f't = {self.times[time_idx]:.2f}s')
            shape_summary = ' | '.join([f'{k}: {v}' for k, v in shape_counts.items() if v > 0])
            shape_text.set_text(shape_summary)
            
            if frame_num % 10 == 0:
                print(f"  Frame {frame_num}/{len(frame_indices)}")
            
            return scatter, time_text, shape_text
        
        anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000/fps, blit=True)
        
        mp4_path = os.path.join(self.ppPath, "bubble_column_animation_deformation.mp4")
        anim.save(mp4_path, writer='ffmpeg', fps=fps, dpi=150)
        plt.close(fig)
        
        print(f"Deformation MP4 saved to: {mp4_path}")
        return mp4_path

def export_for_manim(column, times, ppPath, frame_skip=10):
    """Export simulation data for Manim animation"""
    import pickle
    
    data = {
        'column': {
            'b': column.b,
            'L': column.L,
            'n_nozzles': 6
        },
        'times': times[::frame_skip],
        'bubbles': []
    }
    
    for bubble in column.bubbles:
        bubble_data = {
            'id': bubble.id,
            'D': bubble.D,
            'x_history': bubble.x_history[::frame_skip],
            'y_history': bubble.y_history[::frame_skip],
            'E_history': bubble.E_history[::frame_skip],
            'shape_regime_history': bubble.shape_regime_history[::frame_skip]
        }
        data['bubbles'].append(bubble_data)
    
    output_path = os.path.join(ppPath, 'bubble_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data exported to: {output_path}")
    print(f"To create Manim animation, run: python manim_bubble_animation.py")
    
    return output_path

if __name__ == "__main__":
    ppPath = os.path.join(os.getcwd(), "results")
    
    print("Running bubble column simulation...")
    column, times = run_simulation(tEnd=30.0, dt=1e-2, C_W=0.00025, ppPath=ppPath)
    print(f"Simulation complete. Total bubbles created: {column.bubble_counter}")
    print("Generating plots...")
    plot_results(column, times)
    
    print("\nCreating animations...")
    animator = BubbleAnimator(column, times, ppPath)
    animator.create_animation_mp4(frame_skip=10, fps=10)
    animator.create_animation_with_deformation(frame_skip=10, fps=10)
    
    print("\nExporting data for Manim...")
    export_for_manim(column, times, ppPath, frame_skip=10)
    
    print("Done!")
