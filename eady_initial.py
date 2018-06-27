import matplotlib
matplotlib.use('Agg')

import numpy as np
from periodic_densities import Periodic_density_in_x, sample_rectangle
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri

#RegularMesh = True

#H = 1.e4
#L = 1.e6

#if RegularMesh:
#    npts = 60
#    dx = 1./npts
#    s = np.arange(dx*0.5,1.,dx)
#    s1 = np.arange(-1. + dx*0.5,1.,dx)
#    x,z = np.meshgrid(s1*L,s*H)
#    nx = x.size
#    x = np.reshape(x,(nx,))
#    z = np.reshape(z,(nx,))
#    y = np.array([x,z]).T
#else:
#    N = 500
#    bbox = np.array([0., -0.5, 2., 0.5])
#    Xdens = sample_rectangle(bbox);
#    f0 = np.ones(4)/2;
#   w = np.zeros(Xdens.shape[0]); 
#    T = ma.delaunay_2(Xdens,w);
#    dens = Periodic_density_in_x(Xdens,f0,T,bbox)
#    X = ma.optimized_sampling_2(dens,N,niter=2)
#    x = X[:,0]*L - L
#    z = (X[:,1]+0.5)*H
#    y = np.array([x,z]).T

#Nsq = 2.5e-5
#g = 10.
#f = 1.e-4
#theta0 = 300
#C = 3e-6
#B = 0.255
#B = 1.0e-3* Nsq * theta0 * H / g
#thetap = Nsq*theta0*z/g + B*np.sin(np.pi*(x/L + z/H))
#vg = B*g*H/L/f/theta0*np.sin(np.pi*(x/L + z/H)) - 2*B*g*H/np.pi/L/f/theta0*np.cos(np.pi*x/L)

#plt.scatter(x,z,c=thetap,cmap="plasma")
#plt.show()
#plt.savefig('eady1.png')

#X = vg/f + x
#Z = g*thetap/f/f/theta0
#Y = np.array([X,Z]).T

#thetap = Z*theta0*f**2/g
#sc = plt.scatter(X,Z,c=thetap,cmap="plasma")
#print(thetap.max())
#plt.colorbar(sc)
#plt.savefig('initial.png')
#plt.show()

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
        dens = Periodic_density_in_x(Xdens,f0,T,bbox)
        X = ma.optimized_sampling_2(dens,npts,niter=2)
        x = X[:,0]*L - L
        z = (X[:,1]+0.5)*H
        y = np.array([x,z]).T

    Nsq = 2.5e-5
    g = 10.
    f = 1.e-4
    theta0 = 300
    C = 3e-6
    #B = 0.255
    B = 1.0e-3* Nsq * theta0 * H / g
    thetap = Nsq*theta0*z/g + B*np.sin(np.pi*(x/L + z/H))
    vg = B*g*H/L/f/theta0*np.sin(np.pi*(x/L + z/H)) - 2*B*g*H/np.pi/L/f/theta0*np.cos(np.pi*x/L)

    X = vg/f + x
    Z = g*thetap/f/f/theta0
    Y = np.array([X,Z]).T

    return Y, thetap
    
def eady_OT(Y, bbox, dens, eps_g = 1.e-7,verbose = True):
    H = bbox[3]
    nx = Y[:,0].size
    nu = np.ones(nx)
    nu = (dens.mass() / np.sum(nu)) * nu

    print "mass(nu) = %f" % sum(nu)
    print "mass(mu) = %f" % dens.mass()
    
    w = 0.*Y[:,0]
    Z = Y[:,1]
    mask = Z>0.9*H
    w[mask] = (Z[mask] - 0.9*H)**2
    mask = Z<0.1*H
    w[mask] = (Z[mask] - 0.1*H)**2

    #[f0,m,g0,H] = dens.kantorovich(Y, nu, w)
    #print(m.min())

    w = ma.optimal_transport_2(dens,Y,nu, w0=w, eps_g=1.0e-7,verbose=True)
    return w
    
#bbox = np.array([-L, 0., L, H])
#Xdens = sample_rectangle(bbox)
#f0 = np.ones(4)/(H*2*L)
#rho = np.zeros(Xdens.shape[0])
#T = ma.delaunay_2(Xdens,rho)
#dens = Periodic_density_in_x(Xdens,f0,T,bbox)
#[Yc, w] = eady_OT(Y,dens)
#print(thetap.max())
#sc = plt.scatter(Yc[:,0],Yc[:,1],c=thetap,cmap="plasma")
#plt.colorbar(sc)
#plt.savefig('final.png')
#plt.show()

def forward_euler_sg(Y, dens, tf, bbox, h=1, t0=0.):
    '''
    Function that finds time evolution of semi-geostrophic equations
    using forward Euler method
    
    args:
    
    y0_p initial data in physical co-ordinates
    h time step size
    
    returns:
    
    Y solution in geostrophic co-ordinates at time tf
    '''
    H = bbox[3]
    L = bbox[2]
    g = 10.
    f = 1.e-4
    theta0 = 300
    C = 3e-6

    X = Y[:,0]
    Z = Y[:,1]

    N = int(np.ceil((tf-t0)/h))
    t = np.array([t0 + n*h for n in range(N+1)])
    
    for n in range(1,N+1):
        w = eady_OT(Y, dens)
        [m, Ya] = dens.lloyd(Y,w)
        Z = Z + h*C*g/f/theta0*Ya[:,0]
        X = X + h*C*g/f/theta0*Ya[:,1]
        Y = np.array([X,Z]).T
        
    return Y
