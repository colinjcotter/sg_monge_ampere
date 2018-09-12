import matplotlib
matplotlib.use('Agg')

import os, sys
parentpath = os.path.abspath("..")
if parentpath not in sys.path:
    sys.path.insert(0, parentpath)
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
import periodic_densities as pdx

H = 1.e4          #domain parameters
L = 1.e6
Nsq = 2.5e-5
g = 10.
f = 1.e-4
theta0 = 300.

Heun = True       # use Heun's method for time integration
days = 25         # number of days 
N = 40            # grid points
tstepsize = 1800. # stepsize

#initialise image directory and set path of results directory
if Heun:
    resultsdir = "/scratchcomp04/cvr12/B25_results/Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_heun"
    imgdir = resultsdir+'/laguerre_diagrams_newvg'
    os.mkdir(imgdir)
else:
    resultsdir = "/scratchcomp04/cvr12/B25_results/Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_euler"
    imgdir = resultsdir+'/laguerre_diagrams_newvg'
    os.mkdir(imgdir)

#create triangulation with random vertices - fix for miscolouring
#of pixels on diagonal as suggested by Q.MERIGOT

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

M = int(np.ceil((60*60*24*days)/tstepsize))
n = 0 
count = 0

while n <= M:
    #retrieve results from file and reshape,points
    Y = np.fromfile(resultsdir+'/points_results_'+str(int(tstepsize))+'/points_'+str(int(n))+'.txt',sep = " ")
    l = int(Y.size/2)
    Y = Y.reshape((l,2))
    #weights
    w = np.fromfile(resultsdir+'/weights_results_'+str(int(tstepsize))+'/weights_'+str(int(n))+'.txt',sep=" ")

    #set thetap colour array and reshape for rasterization
    thetap = f*f*theta0*Y[:,1]/g
    thetap = thetap.reshape((l,1))

    A = pdx.periodicinx_rasterization(dens,Y,w,thetap,bbox,1500,1500)

    #rescale colour to thetap
    amax = np.max(A[0]); amin = np.min(A[0])
    tmax = np.max(thetap); tmin = np.min(thetap)
    A[0] = tmin + (tmax-tmin)*(A[0] - amin)/(amax - amin)

    #set vg colour array and reshape for rasterization
    [Yc, m] = dens.lloyd(Y,w)
    vg = f*(Y[:,0]-Yc[:,0])
    #map vg to fundamental domain
    vgf = dens.to_fundamental_domain(np.vstack((vg,vg)))
    vg = vgf[0,:]
    vg = vg.reshape((l,1))
    
    B = pdx.periodicinx_rasterization(dens,Y,w,vg,bbox,1500,1500)

    #rescale colour to vg
    amax = np.max(B[0]); amin = np.min(B[0])
    vgmax = np.max(vg); vgmin = np.min(vg)
    B[0] = vgmin + (vgmax-vgmin)*(B[0] - amin)/(amax - amin)

    #find edges of laguerre cells
    x,y = pdx.periodicinx_draw_laguerre_cells_2(dens,Y,w)

    #plot raserized laguerre diagram,thetap
    plt.figure(figsize=(6,4))
    plt.pcolormesh(A[0].T,vmin = np.min(thetap),vmax = np.max(thetap),cmap = "plasma")
    plt.colorbar()
    plt.axis('off')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(imgdir+'/laguerre_diagram_thetap_'+str(int(count))+'.png')

    #plot raserized laguerre diagram, vg
    plt.figure(figsize=(6,4))
    plt.pcolormesh(B[0].T,vmin = np.min(vg),vmax = np.max(vg),cmap = "plasma")
    plt.colorbar()
    plt.axis('off')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(imgdir+'/laguerre_diagram_vg_'+str(int(count))+'.png')

    #plot laguerre cells
    plt.figure(figsize=(6,4))
    plt.plot(x,y,color=[1,0,0],linewidth=0.5,aa=True)
    plt.xticks(np.array([-1000000,-500000,0,500000,1000000]))
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(imgdir+'/laguerre_tesselation_'+str(int(count))+'.png')

    #plot points in geostrophic space
    plt.figure(figsize=(6,4))
    plt.plot(Y[:,0],Y[:,1],'.')
    plt.xticks(np.array([-1000000,-500000,0,500000,1000000]))
    plt.savefig(imgdir+'/Gpoints_'+str(int(count))+'.png')

    n += 24
    count += 1

