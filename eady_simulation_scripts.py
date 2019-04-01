import numpy as np
import periodic_densities as pdx
import MongeAmpere as ma
import matplotlib.pyplot as plt
import os
from sg_da import initialise_points, eady_OT, forward_euler_sg, heun_sg
from plotting_tools import periodic_laguerre_diagram_to_image

def frontogenesis_timestep(N, days, tstepsize, Heun = None):
    tf = 60*60*24*days #final time
    
    #initialise parameters from SG Eady model
    H = 1.e4
    L = 1.e6
    g = 10.
    f = 1.e-4
    theta0 = 300.
    
    #create new directory to store results
    if Heun:
        newdir = "Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_heun"
    else:
        newdir = "Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_euler"

    os.mkdir(newdir)

    #initialise source density with periodic BCs in x
    bbox = np.array([-L, 0., L, H])
    Xdens = pdx.sample_rectangle(bbox)
    f0 = np.ones(4)/(2*H*L)
    rho = np.zeros(Xdens.shape[0])
    T = ma.delaunay_2(Xdens,rho)
    dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)
    
    #initialise points in geostrophic space
    [Y, thetap] = initialise_points(N, bbox, RegularMesh = False)
    Y = dens.to_fundamental_domain(Y)
    
    if Heun:
        #timestep using Heun's Method
        [Y, w, t] = heun_sg(Y, dens, tf, bbox, newdir, h = tstepsize, add_data = True)
        t.tofile(newdir+'/time.txt',sep=" ",format="%s")
        
    else:
        #timestep using forward euler scheme
        [Y, w, t] = forward_euler_sg(Y, dens, tf, bbox,newdir, h = tstepsize, add_data = True)
        t.tofile(newdir+'/time.txt',sep=" ",format="%s")
        
    return('complete results '+str(days)+'D_'+str(N)+'N_'+str(int(tstepsize)))

def validity_analysis_results(N, days, tstepsize, t0 = 0., Heun = None):
    #initialise parameters
    H = 1.e4
    L = 1.e6
    g = 10.
    f = 1.e-4
    theta0 = 300
    C = 3e-6

    if Heun:
        print('heun')
        datadir = "Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_heun/data"
        resultdir = "Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_heun"
        plotsdir = "Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_heun/plots"
    else:
        print('euler')
        datadir = "Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_euler/data"
        resultdir = "Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_euler"
        plotsdir = "Results_"+str(days)+"D_"+str(N)+"N_"+str(int(tstepsize))+"_euler/plots"
        
    os.mkdir(datadir)
    os.mkdir(plotsdir)
    
    #set up uniform density for domain Gamma
    bbox = np.array([-L, 0., L, H])
    Xdens = pdx.sample_rectangle(bbox)
    f0 = np.ones(4)/2/L/H
    rho = np.zeros(Xdens.shape[0])
    T = ma.delaunay_2(Xdens,rho)
    dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)

    #set up plotting uniform density for domain Gamma
    bbox = np.array([-L, 0., L, H])
    Xdens = pdx.sample_rectangle(bbox)
    npts = 100
    Xrnd = 2*L*(np.random.rand(npts)-0.5)
    Zrnd = H*np.random.rand(npts)
    Xdens = np.concatenate((Xdens, np.array((Xrnd, Zrnd)).T))
    f0 = np.ones(npts+4)/2/L/H
    rho = np.zeros(Xdens.shape[0])
    T = ma.delaunay_2(Xdens,rho)
    pdens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)

    #set final time(t), step size (h) and total number of time steps(N)
    tf = 60*60*24*days
    h = tstepsize
    N = int(np.ceil((tf-t0)/h))
    
    #initialise arrays to store data values
    KEmean = np.zeros(N+1)
    energy = np.zeros(N+1)
    vgmax = np.zeros(N+1)
    PE = np.zeros(N+1)
    KE = np.zeros(N+1)
    rmsv = np.zeros(N+1)
    
    w = np.fromfile(resultdir+'/weights_results_'+str(int(h))+'/weights_0.txt', sep = " ")
    ke = np.zeros((int(w.size),2))
    vg = np.zeros((int(w.size),2))
    t = np.array([t0 + n*h for n in range(N+1)])

    Ndump = 100
    
    for n in range(0,N+1, Ndump):
        print(n,N)
        try:
            Y = np.fromfile(resultdir+'/points_results_'+str(int(h))+'/points_'+str(n)+'.txt', sep = " ")
            w = np.fromfile(resultdir+'/weights_results_'+str(int(h))+'/weights_'+str(n)+'.txt', sep = " ")
            l = int(w.size)
            Y = Y.reshape((l,2))
        except IOError:
            break
        
        #Plots
        C = Y[:,1].reshape((l,1))
        print(C.shape, w.shape)
        img = periodic_laguerre_diagram_to_image(pdens,Y,w,C,bbox,100,100)
        plt.pcolor(img)
        plt.savefig(plotsdir+'/c_'+str(n)+'.jpg')

        plt.clf()
        plt.plot(Y[:,0], Y[:,1], '.')
        plt.savefig(plotsdir+'/Y_'+str(n)+'.jpg')
        plt.clf()

        #calculate centroids and mass of cells
        [Yc, m] = dens.lloyd(Y, w)
        mtile = np.tile(m,(2,1)).T

        #calculate second moments to find KE and maximum of vg
        [m1, I] = dens.moments(Y, w)

        #find kinetic energy, maximum value of vg and total energy
        ke[:,0] = 0.5*f*f*(m*Y[:,0]**2 - 2*Y[:,0]*m1[:,0] + I[:,0])
        vg[:,0] = f*(Y[:,0] - Yc[:,0])
        
        #map back to fundamental domain
        ke = dens.to_fundamental_domain(ke)
        vg = dens.to_fundamental_domain(vg)
        
        E = ke[:,0] - f*f*Y[:,1]*m1[:,1] + 0.5*f*f*H*Y[:,1]*m
        pe = - f*f*Y[:,1]*m1[:,1] + 0.5*f*f*H*Y[:,1]*m
        
        energy[n] = np.sum(E)
        KE[n] = np.sum(ke[:,0])
        KEmean[n] = np.sum(ke[:,0])*float(l)
        rmsv[n] = np.sqrt(KEmean[n])
        PE[n] = np.sum(pe)
        vgmax[n] = np.amax(vg[:,0])
        
    energy.tofile(datadir+'/energy.txt',sep = " ",format = "%s")
    KEmean.tofile(datadir+'/KEmean.txt',sep = " ",format = "%s")
    vgmax.tofile(datadir+'/vgmax.txt',sep = " ",format = "%s")
    t.tofile(datadir+'/time.txt',sep = " ",format = "%s")
    PE.tofile(datadir+'/PE.txt',sep = " ",format = "%s")
    KE.tofile(datadir+'/KE.txt',sep = " ",format = "%s")
    rmsv.tofile(datadir+'/rmsv.txt',sep = " ",format = "%s")
        
    return('complete results '+str(days)+'D_'+str(N)+'N_'+str(int(tstepsize)))
        
        
