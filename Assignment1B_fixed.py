#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:28:09 2019

__authors__ = "Ananda S. Kannan and Niklas Hidman"
__institution__ = "Department of Mechanics and Maritime Sciences, Chalmers University of Technology, Sweden"
__copyright__ = "Copyright 2019, TME160: Multiphase flow"
__license__ = "GPL"
__version__ = "1.0"
__maintainers__ = "Ananda S. Kannan and Niklas Hidman"
__email__ = " ananda@chalmers.se and niklas.hidman@chalmers.se"

####################################################################################################################################
                                        "Py template to simulate a bubble column (in 2D)"

                Assumptions - 
 
                1)	Only 2D motion is studied (motion along the z-axis is assumed to be negligible)
                2)	Bubble starts from rest Vp(t = 0) = 0
                3)	The Continuous phase (water) is assumed to have a (parabolic) laminar velocity profile 
                4)	Only one-way coupling i.e. the background fluid affects the bubble motion and not vice-versa
                5)	5)	Inter-bubble (coalescence and break-up) and bubble-wall interactions (such as collisions etc.) are ignored
####################################################################################################################################
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import sys
np.random.seed(12345)
import csv

simVer = "bubbleColumn"
plt.close("all")
ppPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/TME160/" + simVer
if not os.path.exists(ppPath):
    os.makedirs(ppPath)

statusfile = os.path.join(ppPath, "simulation_status.txt")

"""
####################################################################################################################################
                                            Custom function definitions
####################################################################################################################################

"""

def getBubbleDia(meanDia,stdDevDia):
    """Samples a bubble diameter from a normal distribution"""
    """function I/P's : meanDia (bubble mean diameter)"""
    """function I/P's : stdDevDia (standard deviation of bubble mean diameter)"""
    """function O/P : bubble diameter (which is normally distributed)"""
        
    return np.random.normal(meanDia,stdDevDia)

def injectBubble(meanDia, stdDevDia, bubbleList, bubbleMaxID, bubbleDiaArray, bubbleXpos, bubbleInjectionTimeIndex, injectionXpos, timeIndex):
    """Injects bubbles and stores them in arrays"""
    """Arrays for storing bubble data"""
    """function I/P's : bubbleList (list of all bubbles injected)"""
    """function I/P's : bubbleMaxID (maximum value of the bubble ID among injected bubbles)"""
    """function I/P's : bubbleDiaArray (an array indexed based on bubbleID which contains bubble diameters of all injected bubbles)"""
    """function I/P's : bubbleXpos (an array indexed based on bubbleID which contains the x-positions of all injected bubbles)"""
    """NOTE: All bubbles injected at y = 0"""
    """function I/P's : bubbleInjectionTimeIndex (a list with all injection time indices)"""  
    """injection specific data""" 
    """function I/P's : injectionXpos (injector locations along x-axis )"""
    """function I/P's : timeIndex (time index of the injected bubbles)"""
    """function O/P's : newBubbleID, updatedBubbleList, bubbleDiaArray, bubbleXpos, bubbleInjectionTimeIndex"""   
    
    newBubbleID = bubbleMaxID + 1
    updatedBubbleList = bubbleList.append(newBubbleID)
    try:
       bubbleDiaArray[newBubbleID] = getBubbleDia(meanDia, stdDevDia)
    except:
       sys.exit('!!!You are trying to access an array position that does not exist. Probably the array is too small because n_tot_bubbles is too small. \n n_tot_bubbles is an integer that is used to allocate arrays and must be large enough to fit all bubbles injected during the simulation.')
       
    bubbleXpos[timeIndex, newBubbleID] = injectionXpos
    bubbleInjectionTimeIndex[newBubbleID] = timeIndex
    
    return newBubbleID, updatedBubbleList, bubbleDiaArray, bubbleXpos, bubbleInjectionTimeIndex

def fluidVelandGrad(py, x, b, mul):
    """Calculates the fluid velocity based on a provided laminar (parabolic) profile and calulate the velocity gradient along x-direction"""
    """function I/P's : py (pressure gradient driving the flow in Pa)"""
    """function I/P's : x (x-coordinate in the domain in m)"""
    """function I/P's : b (domain half width along x-dir in m)"""
    """function I/P's : mul (fluid dynamic viscosity in Pa.s)"""
    """function O/P's : fluidVelo = py*x/(2*mul)*(2*b-x); fluidXGrad = py/mul*(b-x)"""
    
    return py*x/(2*mul)*(2*b-x), py/mul*(b-x)

def cl_tomiyama(Re, Eo, sig, rhoL, g): 
        """Calculates the lift co-eff CL based on the Tomiyama model"""
        """function I/P's : Re (Bubble Reynold's number)"""
        """function I/P's : Re (Bubble Eotvo's number)"""
        """function I/P's : sig (Surface tension of the bubble)"""
        """function I/P's : rhoL (Fluid density in Kg/m3)"""
        """function I/P's : g (acceleration due to gravity in m2/s)"""
        """function O/P's : Cl_tom, Eoh (CL i.e. the lift co-efficient; Eo based on bubble horizontal diameter)"""
        
        Dbubble_eq = np.sqrt(Eo*sig/(rhoL*g))
        Dhorisontal = Dbubble_eq*(1+0.163*Eo**(0.757))**(1/3)
        Eoh = rhoL*g*Dhorisontal**2/sig
        feo = 0.00105*Eoh**3 - 0.0159*Eoh**2 - 0.0204*Eoh + 0.474
        if Eoh < 4.0:
            Cl_tom = np.min([ 0.288*np.tanh(0.121*Re), feo])
        elif Eoh < 10.7:
            Cl_tom = feo
        else:
            Eoh=10.7
            Cl_tom = 0.00105*Eoh**3 - 0.0159*Eoh**2 - 0.0204*Eoh + 0.474
            
        return Cl_tom
def f_wall_channel(x, b, d):
      """
      Wall force function for channel geometry
      x: horizontal position of bubble (0 to 2b)
      b: channel half-width
      d: bubble diameter
      Returns: wall force magnitude factor
      """
      x_left_wall = x          # distance from left wall (x=0)
      x_right_wall = 2*b - x   # distance from right wall (x=2b)

      # Repulsion from left wall
      f_left = -d / (x_left_wall**2) if x_left_wall > 0 else 0

      # Repulsion from right wall  
      f_right = d / (x_right_wall**2) if x_right_wall > 0 else 0

      return f_left + f_right

def areaAverageVoidFractionWithinBounds(y_up,y_down,n_bins,timeIndex, b):
    """Calculates a time averaged void fraction within given bounds in the y direction per binned section along the x-direction"""
    """function I/P's : y_up (upper y-bound)"""
    """function I/P's : y_down (lower y-bound)"""
    """function I/P's : n_bins (number of bins to section the x-axis of the domain)"""
    """function I/P's : timeIndex (time index of the injected bubbles)"""
    """function I/P's : b (domain half width along x-dir)"""
    """function O/P : voidFractionPerBin (time averaged void fraction per bin)"""
    
    voidFractionPerBin = np.zeros(n_bins)
    for iBin in range(n_bins):
        xMax = (iBin+1)*2.0*b/n_bins
        xMin = (iBin)*2.0*b/n_bins
        bubblesInBound = list()
        for bubble in range(bubbleMaxID):
            if (bubbleYpos[timeIndex,bubble] > y_down) and (bubbleYpos[timeIndex,bubble] < y_up):
                if (bubbleXpos[timeIndex,bubble] > xMin) and (bubbleXpos[timeIndex,bubble] < xMax):
                    bubblesInBound.append(bubble)

        voidArea = 0
        for bubble in bubblesInBound: 
            voidArea = voidArea + bubbleDia[bubble]**2*np.pi/4.0
            
        totArea = (xMax-xMin)*(y_up-y_down)
        voidFractionPerBin[iBin] = voidArea/totArea
        
    return voidFractionPerBin

"""
####################################################################################################################################
            Bubble column definition: Physical properties and domain description
####################################################################################################################################

"""

b = 0.025
L = 3.0

rhoL = 1000.0
mul = 0.001
sig = 0.073

meanDia = 0.0058
stdDevDia = 0.0015
rhoB = 1.2
massMeanBubble = rhoB*4/3*np.pi*(meanDia/2.0)**3

g=9.82
py = 1

"""
####################################################################################################################################
                        Bubble column definition: time step and injector related settings
####################################################################################################################################

"""
dt = 1e-2
tEnd = 30.0

n_timeSteps = int(np.ceil(tEnd/dt))
times = np.linspace(0,tEnd,n_timeSteps)


massFlowRateTot = 2.4e-6
n_nozzles = 6

massFlowRateNozzle = massFlowRateTot/n_nozzles
bubbleInjectionFrequency = massFlowRateNozzle/massMeanBubble

n_tot_bubbles = int(np.ceil(bubbleInjectionFrequency*tEnd*n_nozzles*2))

"""
####################################################################################################################################
                        Storing bubble data and other related parameters as arrays
####################################################################################################################################

"""
bubbleXpos = np.zeros([n_timeSteps,n_tot_bubbles])
bubbleYpos = np.zeros([n_timeSteps,n_tot_bubbles])
bubbleVelXdir = np.zeros([n_timeSteps,n_tot_bubbles])
bubbleVelYdir = np.zeros([n_timeSteps,n_tot_bubbles])

bubbleDia = np.zeros([n_tot_bubbles])
bubbleInjectionTimeIndex = np.zeros([n_tot_bubbles],dtype=int)
bubbleDeletionTimeIndex = np.zeros([n_tot_bubbles],dtype=int)
aliveBubblesID = list()

"""
####################################################################################################################################
                        Algorithm for the simulation (Forward Euler time integration of the particle equation)
####################################################################################################################################

"""
timeSinceInjection = 1
bubbleMaxID = -1
ti=0

for t in times:

    if timeSinceInjection > 1.0/bubbleInjectionFrequency:
        timeSinceInjection = 0
        for noz in range(n_nozzles):
            injectionPos = 2.0*b/(n_nozzles+1)*(noz+1)
            newBubbleID, updatedBubbleList, bubbleDiaArray, bubbleXpos, bubbleInjectionTimeIndex = injectBubble(meanDia, stdDevDia, aliveBubblesID, bubbleMaxID, bubbleDia, bubbleXpos, bubbleInjectionTimeIndex, injectionPos, ti)
            bubbleMaxID = newBubbleID
    else:
        timeSinceInjection = timeSinceInjection + dt
    
    bubbleRemoveList = list()
    for bubbleID in aliveBubblesID:
        
        if ti == n_timeSteps-1:
            break
        
        D = bubbleDia[bubbleID]
        uBubble = bubbleVelXdir[ti,bubbleID]
        vBubble = bubbleVelYdir[ti,bubbleID]
        
              
        massBubble = rhoB * 4.0/3*np.pi*(D/2.0)**3
        
        """Y-direction (vertical axis)"""   
        
        Vy, dVdx = fluidVelandGrad(py, bubbleXpos[ti,bubbleID], b, mul)
        
        if D < 1.3e-3:
            Vrel = np.sqrt(2*sig/(rhoL*D) + (rhoL - rhoB)*g*D/(2*rhoL))
        else:
            Vrel = vBubble - Vy
        Re = rhoL*np.sqrt(Vrel**2 + uBubble**2)*D/mul
        Eo = (rhoL-rhoB)*g*D**2/sig
        Mo = g*mul**4*(rhoL-rhoB)/(rhoL**2*sig**3)
        We = rhoL*(np.sqrt(Vrel**2 + uBubble**2))**2*D/sig
        
        E = 1.0 / (1.0 + 0.163*Eo**0.757)
        d_horizontal = D / (E**(1/3))
        d_vertical = D * E**(2/3)
        
        if Eo < 0.1:
            shape_regime = "spherical"
        elif Eo < 4.0:
            shape_regime = "ellipsoidal"
        elif Eo < 40.0:
            if We < 3.0:
                shape_regime = "ellipsoidal"
            else:
                shape_regime = "wobbling"
        else:
            shape_regime = "spherical-cap"
        
        contaminated = False
        if contaminated:
            beta = 1
            Cd = beta*max(min(16/Re*(1+0.15*Re**0.687),48/Re),8/3*Eo/(Eo+4))
        else:
            if shape_regime == "spherical":
                mu_ratio = 0.0
                Cd_HR = (24/Re) * (2/3 + mu_ratio) / (1 + mu_ratio)
                Cd = max(Cd_HR, 8/3*Eo/(Eo+4))
            elif shape_regime == "ellipsoidal":
                Cd = (16/Re)*(1 + 0.15*Re**0.687)
            elif shape_regime == "wobbling":
                Cd = max(8/3*Eo/(Eo+4), 0.44)
            else:
                Cd = 8/3*Eo/(Eo+4)
        
        omega = np.array([0, 0, dVdx])
        Vrel_vec = np.array([0, Vrel, 0])
        cross_prod = np.cross(Vrel_vec, omega)
        
        Fb = rhoL * (4/3*np.pi*(D/2)**3) * g
        Fdy = 0.5 * rhoL * Cd * np.pi*D**2/4 * np.sqrt(Vrel**2+uBubble**2)*(Vrel)
        
        Fp = massBubble*rhoL/rhoB*(-g)
        Fg = massBubble*g

        injectionTimeIndex = np.where(bubbleXpos[:,bubbleID] > 0.0)[0][0]
        injTime = times[injectionTimeIndex]
        if (t-injTime) > 0.0:
            mhist = np.sqrt(rhoL*mul*np.pi)*massBubble/(rhoB*D)
            Fhist = mhist*Vrel/np.sqrt(0.5*(t-injTime))
        else:
            Fhist = 0
        
        FtotY = Fb + Fp - Fdy - Fhist - Fg
        totMass = massBubble+(0.5*rhoL*(4/3*np.pi*(D/2)**3))
        
        bubbleVelYdir[ti+1,bubbleID] = vBubble+dt*(FtotY/totMass)
        
        bubbleYpos[ti+1,bubbleID] = bubbleYpos[ti,bubbleID]+dt*bubbleVelYdir[ti+1,bubbleID]
        
        if bubbleYpos[ti+1,bubbleID] > L:
            bubbleRemoveList.append(bubbleID)
            bubbleDeletionTimeIndex[bubbleID] = ti
        
        """X-direction (horizontal axis)"""
        
        Cl = cl_tomiyama(Re, Eo, sig, rhoL, g)
        
        injectionTimeIndex = np.where(bubbleXpos[:,bubbleID] > 0.0)[0][0]
        injTime = times[injectionTimeIndex]
        if (t-injTime) > 0.0:
            mhist = np.sqrt(rhoL*mul*np.pi)*massBubble/(rhoB*D)
            Fhist_x = mhist*(-uBubble)/np.sqrt(0.5*(t-injTime))
        else:
            Fhist_x = 0
        
        FLx = -Cl*rhoL*(np.pi*D**3)/6*(vBubble-Vy)*cross_prod[0]
        Fdx = 0.5*rhoL*Cd*(np.pi*D**2/4)*np.sqrt(Vrel**2+uBubble**2)*(uBubble)

        C_W = 0.00025
        
        min_dist = 5*D
        x_pos = bubbleXpos[ti, bubbleID]
        x_left = max(x_pos, min_dist)
        x_right = max(2*b - x_pos, min_dist)
        
        if x_left < min_dist or x_right < min_dist:
            f_w_left = -D / (min_dist**2) if x_pos < min_dist else 0
            f_w_right = D / (min_dist**2) if (2*b - x_pos) < min_dist else 0
        else:
            f_w_left = -D / (x_left**2)
            f_w_right = D / (x_right**2)
        
        f_w = f_w_left + f_w_right
        F_W = C_W * f_w * 0.5 * rhoL * (np.pi * D**2 / 4) * np.sqrt(Vrel**2 + uBubble**2)**2

        FtotX = FLx - Fdx - Fhist_x + F_W
        
        if not np.isfinite(FtotX):
            FtotX = 0.0
        
        bubbleVelXdir[ti+1,bubbleID] = uBubble+dt*(FtotX/totMass)
        
        bubbleXpos[ti+1,bubbleID] = bubbleXpos[ti,bubbleID]+dt*bubbleVelXdir[ti+1,bubbleID]
        
        if (bubbleXpos[ti+1,bubbleID] < D/2.0) and (bubbleVelXdir[ti,bubbleID] < 0.0):
            bubbleVelXdir[ti,bubbleID] = 0.0
            bubbleXpos[ti+1,bubbleID] = bubbleXpos[ti,bubbleID]
        elif (bubbleXpos[ti+1,bubbleID] > (2*b-D/2.0)) and (bubbleVelXdir[ti,bubbleID] > 0.0):
            bubbleVelXdir[ti,bubbleID] = 0.0
            bubbleXpos[ti+1,bubbleID] = bubbleXpos[ti,bubbleID]
    
        with open(statusfile, 'w') as f:
            f.write(f"{t:.4f},{bubbleID},{bubbleXpos[ti,bubbleID]:.6f},{bubbleYpos[ti,bubbleID]:.6f},{uBubble:.6f},{vBubble:.6f},{D:.6f},{Vrel:.6f},{Re:.2f},{Cd:.6f},{Fb:.8f},{Fdy:.8f},{Fhist:.8f},{FtotY:.8f},{FLx:.8f},{Fdx:.8f},{FtotX:.8f}\n")
    
    for bubble in bubbleRemoveList:
        aliveBubblesID.remove(bubble)
        
    ti=ti+1

    
"""
####################################################################################################################################
                        Post-processing the bubble data: relevant plots
####################################################################################################################################

"""
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "serif"

fig = plt.figure(figsize=(15, 5))
fig.suptitle('Bubble Column Simulation\nForces: Buoyancy, Drag, History, Pressure Gradient, Lift (Tomiyama), Wall Force (Cw = 0.00001)', 
             fontsize=12, fontweight='bold')
gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3, top=0.88)

VyToPlot, dVdxToPlot = fluidVelandGrad(py, np.linspace(0,2*b,100), b, mul) 

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlabel('x-coord')
ax1.set_ylabel('y-coord')
for bubble in range(bubbleMaxID):
    tStart = bubbleInjectionTimeIndex[bubble]+1
    tEnd = bubbleDeletionTimeIndex[bubble]-1
    ax1.plot(bubbleXpos[tStart:tEnd,bubble],bubbleYpos[tStart:tEnd,bubble],color=cm.jet(bubbleDia[bubble]/np.max(bubbleDia)))

ax1_twin = ax1.twinx()
ax1_twin.set_ylabel('Velocity profile')
ax1_twin.plot(np.linspace(0,2*b,100),VyToPlot,'--', color='red', linewidth=2)
ax1.set_title('Bubble trajectory (colored by bubble size)')
ax1.grid(True)

binsInXdir = 10
dy = 0.02

avVoidFracArea1 = np.zeros([n_timeSteps,int(binsInXdir)])
avVoidFracArea2 = np.zeros([n_timeSteps,int(binsInXdir)])
avVoidFracArea3 = np.zeros([n_timeSteps,int(binsInXdir)])

for i in range(n_timeSteps):
    avVoidFracArea1[i,:] = areaAverageVoidFractionWithinBounds(0.3,0.3-dy,binsInXdir,i,b)
    avVoidFracArea2[i,:]  = areaAverageVoidFractionWithinBounds(1,1-dy,binsInXdir,i,b)
    avVoidFracArea3[i,:]  = areaAverageVoidFractionWithinBounds(2,2-dy,binsInXdir,i,b)
    
startAverTimeIndex = int(0.3 * n_timeSteps)
timeAverageOverBins1 = np.mean(avVoidFracArea1[startAverTimeIndex:-1,:],axis=0)
timeAverageOverBins2 = np.mean(avVoidFracArea2[startAverTimeIndex:-1,:],axis=0)
timeAverageOverBins3 = np.mean(avVoidFracArea3[startAverTimeIndex:-1,:],axis=0)

ax2 = fig.add_subplot(gs[0, 1])
x_pos_bins = [i*2*b/binsInXdir+2*b/(2*binsInXdir) for i in range(binsInXdir)]
ax2.plot(x_pos_bins,timeAverageOverBins1,'-*')
ax2.plot(x_pos_bins,timeAverageOverBins2,'-*')
ax2.plot(x_pos_bins,timeAverageOverBins3,'-*')
ax2.set_xlabel('x-pos')
ax2.set_ylabel('time av void fraction')
ax2.legend(['y = 0.3m','y = 1.0m','y = 2.0m'])
ax2.set_title('Area avg void fraction')
ax2.grid(True)

ax3 = fig.add_subplot(gs[0, 2])
timeIndexToPlot = int(0.5 * n_timeSteps)
for bubble in range(bubbleMaxID):
    if (bubbleYpos[timeIndexToPlot,bubble] > 0.0):
        ax3.plot(bubbleXpos[timeIndexToPlot,bubble],bubbleYpos[timeIndexToPlot,bubble],'o',color=cm.jet(bubbleDia[bubble]/np.max(bubbleDia)))
ax3.set_ylim([0,L])
ax3.set_xlabel('x-pos') 
ax3.set_ylabel('y-pos') 
ax3.set_title('Bubble pos snapshot')
ax3.grid(True)

figName = "Bubble_column_results.png"
plt.savefig(os.path.join(ppPath, figName), dpi=250, bbox_inches='tight')
plt.show()
