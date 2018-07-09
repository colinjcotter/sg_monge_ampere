import matplotlib
matplotlib.use('Agg')

import numpy as np
from periodic_densities import Periodic_density_in_x, sample_rectangle,periodicinx_draw_laguerre_cells_2
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from PIL import Image

# source: uniform measure on the square with sidelength 1
bbox = [0.,0.,1.,1.]
X = np.array([[0, 0],
              [1, 0],
              [1, 1],
              [0, 1]], dtype=float);
#T = ma.delaunay_2(X,np.zeros(4));
#mu = np.ones(4);
#dens = ma.Density_2(X,mu,T);
#print "mass=%g"%dens.mass()

Xdens = sample_rectangle(bbox)
f0 = np.ones(4)
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = Periodic_density_in_x(Xdens,f0,T,bbox)
print "mass=%g"%dens.mass()

# target is a random set of points, with random weights
N = 6;
Y = np.random.rand(N,2)/2;
nu = 10+np.random.rand(N);
nu = (dens.mass() / np.sum(nu)) * nu;

# target is a random set of points, with random weights
N = 6;
Y = np.random.rand(N,2);
nu = np.ones(N);
nu = (dens.mass() / np.sum(nu)) * nu;

print "mass(nu) = %f" % sum(nu)
print "mass(mu) = %f" % dens.mass()
 
w = ma.optimal_transport_2(dens,Y,nu)

C = np.random.rand(N,1)

N = Y.shape[0]
Y0 = dens.to_fundamental_domain(Y)
x0 = dens.u[0]
v = np.array([[0,0], [x0,0], [-x0,0]])
Yf = np.zeros((3*N,2))
wf = np.hstack((w,w,w))
Cf = np.vstack((C,C,C))
for i in xrange(0,3):
    Nb = N*i
    Ne = N*(i+1)
    Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

img = ma.laguerre_diagram_to_image(dens,Yf,wf,Cf,bbox,1000,1000)
print(type(img))
print(img.shape)
img.tofile("laguerre_diagram.txt",sep=" ",format="%s")
#print(type(img))
#img.save("img1.png","png")

[E,x,y] = periodicinx_draw_laguerre_cells_2(dens,Y,w)
Y.tofile('points_data.txt',sep=" ",format="%s")
x.tofile('x_data.txt',sep=" ",format="%s")
y.tofile('y_data.txt',sep=" ",format="%s")
plt.plot(Y[:,0],Y[:,1],'.')
plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)
plt.savefig('periodic_plot.png')
