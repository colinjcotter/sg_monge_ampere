import matplotlib
matplotlib.use('Agg')

import numpy as np
from periodic_densities import Periodic_density_in_x, sample_rectangle,periodicinx_draw_laguerre_cells_2
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# source: uniform measure on the square with sidelength 1
bbox = [0.,0.,1.,1.]
Xdens = sample_rectangle(bbox)
f0 = np.ones(4)
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = Periodic_density_in_x(Xdens,f0,T,bbox)
print "mass=%g"%dens.mass()

# target is a random set of points, with random weights
N = 6;
Y = np.random.rand(N,2);
nu = np.ones(N);
nu = (dens.mass() / np.sum(nu)) * nu;

print "mass(nu) = %f" % sum(nu)
print "mass(mu) = %f" % dens.mass()
 
w = ma.optimal_transport_2(dens,Y,nu)

#x,y = periodicinx_draw_laguerre_cells_2(dens,Y,w)

#[mf,Yf,If] = dens.moments(Y,w)
#Yf /= np.tile(mf,(2,1)).T
[Yc,m] = dens.lloyd(Y,w)
[E,x,y] = periodicinx_draw_laguerre_cells_2(dens,Y,w)
print(E.shape)
print(x.shape)
print(y.shape)
#print(Yc)
#print(mf)
#print(m)
x.tofile('x_data.txt',sep=" ",format="%s")
y.tofile('y_data.txt',sep=" ",format="%s")
E.tofile('E_data.txt',sep=" ",format="%s")
Y.tofile('Y1_data.txt',sep=" ",format="%s")
Yc.tofile('Yc_data.txt',sep=" ",format="%s")


plt.plot(Y[:,0],Y[:,1],'.')
plt.plot(Yc[:,0],Yc[:,1],'.')
plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)
plt.savefig('periodic_plot.png')

