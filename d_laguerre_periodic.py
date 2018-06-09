import sys
import os
sys.path.append('..');

import matplotlib
matplotlib.use('Agg')
import MongeAmpere as ma
from periodic_densities import Periodic_density_in_x, sample_rectangle
import numpy as np
import matplotlib.pyplot as plt
import pdb

RegularMesh = False

if RegularMesh:
    L = 0.5
    H = 1.
    N = 10
    dx = 1./N
    s = np.arange(dx*0.5,1.,dx)
    s1 = np.arange(-1. + dx*0.5,1.,dx)
    x,z = np.meshgrid(s1*L,s*H)
    nx = x.size
    x = np.reshape(x,(nx,))
    z = np.reshape(z,(nx,))
    Y = np.array([x,z]).T
    Y[:.1] *= 2.
    X = Y[:,0]
    Z = Y[:,1]
    
else:
#Random points in domain D = [0,2]x[0,2]
    bbox = [0.,0.,1.,1.]
    N = 10
    Y = np.random.rand(N,2)
    Y[:,1] *= 2.
    X = Y[:,0]
    Z = Y[:,1]
    
#pdb.set_trace()
nx = X.size
bbox = [0.,0.,1.,1.]

'''bbox in periodic density case is bbox = [x0,y0,xf,yf]'''
plt.plot(Y[:,0],Y[:,1],'.')
plt.savefig('initial_points.png')

#pdb.set_trace()
Xdens = sample_rectangle(bbox)
f0 = np.ones(4)
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = Periodic_density_in_x(Xdens,f0,T,bbox)
nu = np.ones(nx)
nu = (dens.mass() / np.sum(nu)) * nu
w = 0.*Y[:,0]

'''
#map points to fundamental domain
Y1 = Periodic_density_in_x.to_fundamental_domain(dens,Y)
plt.plot(Y1[:,0],Y1[:,1],'.')
plt.savefig('fd_points.png')
'''

mask = Z>0.9
w[mask] = (Z[mask] - 0.9)**2
mask = Z<0.1
w[mask] = (Z[mask] - 0.1)**2

w = ma.optimal_transport_2(dens,Y,nu,w0=w,verbose=True)

def periodicinx_draw_laguerre_cells_2(pxdens,Y,w):
    # draw laguerre cells when boundary is periodic in
    # x direction
    N = Y.shape[0]
    Y0 = pxdens.to_fundamental_domain(Y)
    x = pxdens.u[0]
    y = pxdens.u[1]
    v = np.array([[0,0], [x,0], [-x,0]])
    Yf = np.zeros((3*N,2))
    wf = np.hstack((w,w,w))
    for i in xrange(0,3):
        Nb = N*i
        Ne = N*(i+1)
        Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))
    E = pxdens.restricted_laguerre_edges(Yf,wf)
    nan = float('nan')
    N = E.shape[0]
    x = np.zeros(3*N)
    y = np.zeros(3*N)
    a = np.array(range(0,N))
    x[3*a] = E[:,0]
    x[3*a+1] = E[:,2]
    x[3*a+2] = nan
    y[3*a] = E[:,1]
    y[3*a+1] = E[:,3]
    y[3*a+2] = nan
    return x,y

pxx, pxy = periodicinx_draw_laguerre_cells_2(dens,Y,w)
plt.figure()
plt.plot(pxx,pxy,color = [1,0,0],linewidth=1,aa=True)
plt.plot(Y[:,0],Y[:,1],'.',color = [0,1,0])
plt.savefig('periodicinx.png')
