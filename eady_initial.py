import matplotlib
matplotlib.use('Agg')

import numpy as np
from periodic_densities import Periodic_density_in_x, sample_rectangle
import MongeAmpere as ma
import matplotlib.pyplot as plt

RegularMesh = True

H = 1.e4
L = 1.e6

if RegularMesh:
    npts = 40
    dx = 1./npts
    s = np.arange(dx*0.5,1.,dx)
    s1 = np.arange(-1. + dx*0.5,1.,dx)
    x,z = np.meshgrid(s1*L,s*H)
    nx = x.size
    x = np.reshape(x,(nx,))
    z = np.reshape(z,(nx,))
    Y = np.array([x,z]).T
else:
    N = 10000
    bbox = np.array([0., -0.5, 2., 0.5])
    Xdens = sample_rectangle(bbox);
    f0 = np.ones(4);
    w = np.zeros(Xdens.shape[0]); 
    T = ma.delaunay_2(Xdens,w);
    dens = Periodic_density_in_x(Xdens,f0,T,bbox)
    X = ma.optimized_sampling_2(dens,N,niter=2)
    x = X[:,0]*L - L
    z = (X[:,1]+0.5)*H

Nsq = 2.5e-5
g = 10.
f = 1.e-4
theta0 = 300
C = 3e-6
#B = 0.255
B = 1.0e-3* Nsq * theta0 * H / g
thetap = Nsq*theta0*z/g + B*np.sin(np.pi*(x/L + z/H))
vg = B*g*H/L/f/theta0*np.sin(np.pi*(x/L + z/H)) - 2*B*g*H/theta0/L/f/theta0*np.cos(np.pi*x/L)

X = vg/f + x
Z = g*thetap/f/f/theta0

Y = np.array([X,Z]).T
bbox = np.array([-L, 0., L, H])
Xdens = sample_rectangle(bbox)
f0 = np.ones(4)
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = Periodic_density_in_x(Xdens,f0,T,bbox)
nx = X.size
nu = np.ones(nx)
nu = (dens.mass() / np.sum(nu)) * nu
w = 0.*Y[:,0]

mask = Z>0.9*H
w[mask] = (Z[mask] - 0.9*H)**2
mask = Z<0.1*H
w[mask] = (Z[mask] - 0.1*H)**2

[f,m,g,H] = dens.kantorovich(Y, nu, w)
print(m.min())

w = ma.optimal_transport_2(dens,Y,nu, w0=w, eps_g=5.0e-2,verbose=True)
Y0, m = dens.lloyd(Y, w)
print("computed moments")

import matplotlib.pyplot as plt
import matplotlib.tri as tri
triang = tri.Triangulation(Y0[:,0], Y0[:,1])
plt.figure()
plt.tripcolor(triang, Z*theta0*f**2/g, shading='flat')
plt.savefig('eady_init.png')
