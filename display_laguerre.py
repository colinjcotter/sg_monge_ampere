import sys
import os
sys.path.append('..');

import matplotlib
matplotlib.use('Agg')
import MongeAmpere as ma
import numpy as np
import matplotlib.pyplot as plt


def draw_laguerre_cells(dens,Y,w):
    E = dens.restricted_laguerre_edges(Y,w)
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
    #plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)
    return x,y

#initialise source continuous density
dens = ma.Density_2(np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]]))

#initialise points
N = 10
Y = np.random.rand(N,2)

#initialise discrete target density
nu = np.ones(N);
nu = (dens.mass() / np.sum(nu)) * nu;

#draw laguerre cells with zero weights
x,y = draw_laguerre_cells(dens,Y,np.zeros(N))
plt.figure()
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(Y[:,0],Y[:,1],'.')
plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)
plt.savefig('laguerre_diagram_0w.png')

#draw laguerre cells with weights calulated from OT solver
w = ma.optimal_transport_2(dens,Y,nu)
x,y = draw_laguerre_cells(dens,Y,w)
plt.figure()
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(Y[:,0],Y[:,1],'.')
plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)
plt.savefig('laguerre_diagram_OTw.png')
