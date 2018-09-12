import matplotlib
matplotlib.use('Agg')

import numpy as np
import periodic_densities as pdx
import MongeAmpere as ma
import os
from sg_da import initialise_points, eady_OT, forward_euler_sg, heun_sg
import cProfile

pr = cProfile.Profile()

N = 30
tf = 60*60
tstepsize = 1800
Heun = True

#initialise parameters from SG Eady model
H = 1.e4
L = 1.e6
g = 10.
f = 1.e-4
theta0 = 300.

#initialise source density with periodic BCs in x
bbox = np.array([-L, 0., L, H])
Xdens = pdx.sample_rectangle(bbox)
f0 = np.ones(4)/(2*H*L)
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)


#initialise points in geostrophic space
[Y, thetap] = initialise_points(N, bbox, RegularMesh = True)
Y = dens.to_fundamental_domain(Y)

if Heun:
    #timestep using Heun's Method
    print('Heun '+str(N)+'N')
    pr.enable()
    [Y, w, t] = heun_sg(Y, dens, tf, bbox, h = tstepsize)
    pr.disable()
else:
    #timestep using forward euler scheme
    print('Euler '+str(N)+'N')
    pr.enable()
    [Y, w, t] = forward_euler_sg(Y, dens, tf, bbox, h = tstepsize)
    pr.disable()

pr.print_stats(sort='time')

