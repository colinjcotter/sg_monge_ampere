import sys
import os
sys.path.append('..');

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
    plt.plot(Y[:,0],Y[:,1],'.')
    plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)

dens = ma.Density_2(np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]]))
N = 400
s = np.arange(0.,1.,0.01)
x,y = np.meshgrid(s,s)
nx = x.size
x = np.reshape(x,(nx,))
y = np.reshape(y,(nx,))
Y = np.array([x,y]).T
Y[:,1] += -(0.5-Y[:,1])*((Y[:,0]-0.5)**2 - 3./8.)
#plt.plot(Y[:,0],Y[:,1],'.')
nu = np.ones(nx)
nu = (dens.mass() / np.sum(nu)) * nu;
w = 0.*Y[:,0]
w = ma.optimal_transport_2(dens,Y,nu, verbose=True)
draw_laguerre_cells(dens,Y,w)
plt.show()

