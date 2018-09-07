import matplotlib
matplotlib.use('Agg')

import os, sys
parentpath = os.path.abspath("..")
if parentpath not in sys.path:
    sys.path.insert(0, parentpath)
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import periodic_densities as pdx
from PIL import Image

H = 1.e4          #domain parameters
L = 1.e6
Nsq = 2.5e-5
g = 10.
f = 1.e-4
theta0 = 300.

Heun = True      # use Heun's method for time integration
days = 25         # number of days 
N = 40            # grid points
tstepsize = 1800. # stepsize

#initialise image directory and set path of results directory
if Heun:
    resultsdir = "/scratchcomp04/cvr12/B25_results/Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_heun"
    imgdir = resultsdir+'/laguerre_diagrams'
    #os.mkdir(imgdir)
else:
    resultsdir = "/scratchcomp04/cvr12/B25_results/Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_euler"
    imgdir = resultsdir+'/laguerre_diagrams'
    os.mkdir(imgdir)

colour = False

Tdomain = np.array([[-L ,0.],[-L, H],[L,H],[L,0.]])
Tpoints = np.random.rand(20,2)
Tpoints[:,0] = Tpoints[:,0]*2*L - L
Tpoints[:,1] = Tpoints[:,1]*H
Tri = np.vstack((Tdomain,Tpoints))

#initialise periodic density
bbox = np.array([-L, 0., L, H])
T = ma.delaunay_2(Tri)
f0 = np.ones(24)/2/L/H

dens = pdx.Periodic_density_in_x(Tri,f0,T,bbox)


#M = int(np.ceil((60*60*24*days)/tstepsize))
M = 1
n = 0
count = 0

while n <= M:
    Y = np.fromfile(resultsdir+'/points_results_'+str(int(tstepsize))+'/points_'+str(int(n))+'.txt',sep = " ")

    l = int(Y.size/2)
    Y = Y.reshape((l,2))

    w = np.fromfile(resultsdir+'/weights_results_'+str(int(tstepsize))+'/weights_'+str(int(n))+'.txt',sep=" ")

    #set colour array and reshape for rasterization
    thetap = f*f*theta0*Y[:,1]/g
    thetap = thetap.reshape((l,1))
    thetap = (thetap - min(thetap))/(max(thetap)-min(thetap))

    A = pdx.periodicinx_rasterization(dens,Y,w,thetap,bbox,1500,1500)

    x,y = pdx.periodicinx_draw_laguerre_cells_2(dens,Y,w)
    
    # plt.figure(figsize=(6,4))
    # plt.pcolormesh(A[0].T,cmap = "plasma")
    # plt.colorbar()
    # plt.axis('off')
    # plt.savefig(imgdir+'/laguerre_diagram_'+str(int(count))+'.png')

    # plt.figure(figsize=(6,4))
    # plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)
    # plt.savefig(imgdir+'/laguerre_tesselation_'+str(int(count))+'.png')

    plt.figure(figsize=(6,4))
    plt.plot(Y[:,0],Y[:,1],'.')
    plt.xticks(np.array([-1000000,-500000,0,500000,1000000]))
    plt.savefig(imgdir+'/Gpoints_'+str(int(count))+'.png')

    n += 24
    count += 1

