import matplotlib
matplotlib.use('Agg')

import numpy as np
from periodic_densities import Periodic_density_in_x, sample_rectangle
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

def initialise_points(N, bbox, RegularMesh = False):
    '''Function to initialise a mesh over the domain [-L,L]x[0,H]
    and transform to geostrophic coordinates

    args:
    N: number of grid points in z direct, total number of points
      will be 2*N*N

    RegularMesh: controls whether the mesh is regular or an optimised
                 sample of random points in the domain 

    returns:
    
    A numpy array of coordinates of points in geostrophic space
    '''
    H = bbox[3]
    L = bbox[2]

    if RegularMesh:
        npts = N
        dx = 1./npts
        s = np.arange(dx*0.5,1.,dx)
        s1 = np.arange(-1. + dx*0.5,1.,dx)
        x,z = np.meshgrid(s1*L,s*H)
        nx = x.size
        x = np.reshape(x,(nx,))
        z = np.reshape(z,(nx,))
        y = np.array([x,z]).T
    else:
        npts = 2*N*N
        bbox = np.array([0., -0.5, 2., 0.5])
        Xdens = sample_rectangle(bbox);
        f0 = np.ones(4)/2;
        w = np.zeros(Xdens.shape[0]); 
        T = ma.delaunay_2(Xdens,w);
        Sdens = Periodic_density_in_x(Xdens,f0,T,bbox)
        X = ma.optimized_sampling_2(Sdens,npts,niter=5)
        x = X[:,0]*L - L
        z = (X[:,1]+0.5)*H
        y = np.array([x,z]).T

    Nsq = 2.5e-5
    g = 10.
    f = 1.e-4
    theta0 = 300
    C = 3e-6
    B = 0.255
    #B = 1.0e-3* Nsq * theta0 * H / g
    thetap = Nsq*theta0*z/g + B*np.sin(np.pi*(x/L + z/H))
    vg = B*g*H/L/f/theta0*np.sin(np.pi*(x/L + z/H)) - 2*B*g*H/np.pi/L/f/theta0*np.cos(np.pi*x/L)
    
    X = vg/f + x
    Z = g*thetap/f/f/theta0
    Y = np.array([X,Z]).T
    return Y, thetap
    
def eady_OT(Y, bbox, dens, eps_g = 1.e-7,verbose = False):
    H = bbox[3]
    nx = Y[:,0].size
    nu = np.ones(nx)
    nu = (dens.mass() / np.sum(nu)) * nu
    
    w = 0.*Y[:,0]
    Z = Y[:,1]
    mask = Z>0.9*H
    w[mask] = (Z[mask] - 0.9*H)**2
    mask = Z<0.1*H
    w[mask] = (Z[mask] - 0.1*H)**2

    w = ma.optimal_transport_2(dens,Y,nu, w0=w, eps_g=1.0e-5,verbose=False)
    return w

def forward_euler_sg(Y, dens, tf, bbox, h=1800, t0=0., add_data=False):
    '''
    Function that finds time evolution of semi-geostrophic equations
    using forward Euler method
    
    args:
    
    Y initial data in Geostrophic co-ordinates
    h time step size
    
    returns:
    
    Y numpy array solution in physical co-ordinates at time tf
    '''
    os.mkdir('timestep_results_9D')
    H = bbox[3]
    L = bbox[2]
    Nsq = 2.5e-5
    g = 10.
    f = 1.e-4
    B = 0.255
    theta0 = 300
    C = 3e-6

    N = int(np.ceil((tf-t0)/h))
    
    if add_data:
        vg = np.zeros(N)
        thetap = np.zeros(N+1)
        energy = np.zeros(N+1)
        t = np.array([t0 + n*h for n in range(N+1)])
        t.tofile('timestep_results_9D/time.txt',sep=" ",format="%s")
        
    for n in range(1,N+1):
        w = eady_OT(Y, bbox, dens)
        [Ya, m] = dens.lloyd(Y, w)
        
        if add_data:
            #calculate second moments to find energy and RMSV
            I = dens.second_moment(Y, w)
            thetap = Y[:,1]*f*f*theta0/g  
            rmsv = f**2*(m*Y[:,0]**2 - 2*Y[:,0]*Ya[:,0] + I[:,0])
            E = 0.5*rmsv - f*f*Y[:,1]*Ya[:,1] + 0.5*f*f*H*Y[:,1]*m
            energy[n-1] = np.sum(E)
            vg[n-1] = np.amax(rmsv)

        #timestep using euler method
        Y[:,1] = Y[:,1] + h*C*g/f/theta0*(Y[:,0] - Ya[:,0])
        Y[:,0] = Y[:,0] + h*C*g/f/theta0*(Ya[:,1] - H*np.ones(Ya[:,1].size)/2.)
        Y = dens.to_fundamental_domain(Y)
        
    Y.tofile('timestep_results_9D/Gpoints_'+str(int(t[N]))+'.txt',sep=" ",format="%s")
    w = eady_OT(Y, bbox, dens)
    w.tofile('timestep_results_9D/weights_'+str(int(t[N]))+'.txt',sep=" ",format="%s")

    if add_data:
        I = dens.second_moment(Y,w)
        thetap = Y[:,1]*f*f*theta0/g
        thetap.tofile('timestep_results_9D/thetap.txt',sep = " ",format="%s")
        rmsv = f**2*(m*Y[:,0]**2 - 2*Y[:,0]*Ya[:,0] + I[:,0])
        E = 0.5*rmsv - f*f*Y[:,1]*Ya[:,1] + 0.5*f*f*H*Y[:,1]*m
        energy[N] = np.sum(E)
        vg[N] = np.amax(rmsv)
        energy.tofile('timestep_results_9D/energy.txt',sep = " ",format="%s")
        vg.tofile('timestep_results_9D/vg.txt',sep = " ",format="%s")

    #find centroids of the cells (physical points)   
    [Y, m] = dens.lloyd(Y,w)
    Y = dens.to_fundamental_domain(Y)
    return Y, w

def heun_sg(Y, dens, tf, bbox, h=1800, t0=0.):
    '''
    Function that finds time evolution of semi-geostrophic equations
    using Heun's order 2 method
    
    args:
    
    y0_p initial data in physical co-ordinates
    h time step size
    
    returns:
    
    Y numpy array solution in geostrophic co-ordinates at time tf
    '''
    H = bbox[3]
    L = bbox[2]
    g = 10.
    f = 1.e-4
    theta0 = 300
    C = 3e-6

    N = int(np.ceil((tf-t0)/h))
    
    t = np.array([t0 + n*h for n in range(N+1)])

    ny = Y[:,0].size
    Yn = np.zeros((ny,2))
    
    for n in range(1,N+1):
        print(n)
        w = eady_OT(Y, bbox, dens)
        [Ya, m] = dens.lloyd(Y,w)
        Yn[:,1] = Y[:,1] + h*C*g/f/theta0*(Y[:,0] - Ya[:,0])
        Yn[:,0] = Y[:,0] + h*C*g/f/theta0*(Ya[:,1] - H*np.ones(Ya[:,1].size)/2.)
        w = eady_OT(Yn, bbox, dens)
        [Yb, m] = dens.lloyd(Yn,w)
        Y[:,1] = Y[:,1] + 0.5*h*C*g/f/theta0*(Y[:,0] - Ya[:,0]) + 0.5*h*C*g/f/theta0*(Yn[:,0] - Yb[:,0])
        Y[:,0] = Y[:,0] + 0.5*h*C*g/f/theta0*(Ya[:,1] - H*np.ones(Ya[:,1].size)/2.) + 0.5*h*C*g/f/theta0*(Yb[:,1] - H*np.ones(Yb[:,1].size)/2.)
        Y = dens.to_fundamental_domain(Y)

    Y.tofile('Gpoints_'+str(n)+'.txt',sep=" ",format="%s")
    w = eady_OT(Y, bbox, dens)
    w.tofile('weights_'+str(n)+'.txt',sep=" ",format="%s")
    [Y, m] = dens.lloyd(Y,w)
    Y = dens.to_fundamental_domain(Y)
    return Y, w
