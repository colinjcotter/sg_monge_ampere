import matplotlib
matplotlib.use('Agg')

import numpy as np
from periodic_densities import Periodic_density_in_x, sample_rectangle
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from eady_initial import initialise_points, eady_OT, forward_euler_sg, heun_sg

add_data = True
Heun = True
days = 25

#initialise number of grid points and final time
N = 40
tf = 60*60*24*days
tstepsize = 1800.

#initialise parameters from SG Eady model
H = 1.e4
L = 1.e6
g = 10.
f = 1.e-4
theta0 = 300.

#create new directory to store results
newdir = "Results_"+str(days)+"D_"+str(N)+"N_euler"
os.mkdir(newdir)

#initialise source density with periodic BCs in x
bbox = np.array([-L, 0., L, H])
Xdens = sample_rectangle(bbox)
f0 = np.ones(4)/(2*H*L)
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = Periodic_density_in_x(Xdens,f0,T,bbox)

#initialise points in geostrophic space
[Y, thetap] = initialise_points(N, bbox, RegularMesh = False)
Y = dens.to_fundamental_domain(Y)
Y.tofile(newdir+'/points_0.txt',sep=" ",format="%s")
thetap.tofile(newdir+'/thetap_init.txt',sep=" ",format="%s")

if Heun:
    #timestep using Heun's Method
    [Y, w, t] = heun_sg(Y, dens, tf, bbox, add_data = True)
    Y.tofile(newdir+'/Gpoints_final.txt',sep=" ",format="%s")
    w.tofile(newdir+'/weights_final.txt',sep=" ",format="%s")
    t.tofile(newdir+'/time.txt',sep=" ",format="%s")
    thetap = Y[:,1]*f*f*theta0/g
    thetap.tofile(newdir+'/thetap_final.txt',sep=" ",format="%s")

else:
    #timestep using forward euler scheme
    [Y, w, t] = forward_euler_sg(Y, dens, tf, bbox, add_data = True)
    Y.tofile(newdir+'/Gpoints_final.txt',sep=" ",format="%s")
    w.tofile(newdir+'/weights_final.txt',sep=" ",format="%s")
    t.tofile(newdir+'/time.txt',sep=" ",format="%s")
    thetap = Y[:,1]*f*f*theta0/g
    thetap.tofile(newdir+'/thetap_final.txt',sep=" ",format="%s")

    
