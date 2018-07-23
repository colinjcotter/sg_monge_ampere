import matplotlib
matplotlib.use('Agg')

import numpy as np
import periodic_densities as pdx
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from PIL import Image

N = 10000
L = 1.e6
H = 1.e4

bbox = np.array([-L, 0., L, H])
Xdens = pdx.sample_rectangle(bbox)
f0 = np.ones(4)/(2*L*H)
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)

Y = np.random.rand(N,2)
xY[:,0] = Y[:,0]*2*L - L
Y[:,1] *= H
#print(Y)

print(Y.shape)
nu = np.ones(Y[:,0].size);
nu = (dens.mass() / np.sum(nu)) * nu

w = ma.optimal_transport_2(dens, Y,nu , verbose = True)

#C = Y[:,1].reshape((N,1))
#print(C)
Nsq = 2.5e-5
g = 10.
f = 1.e-4
theta0 = 300
C = 3e-6
B = 0.255
C = Nsq*theta0*Y[:,1]/g + B*np.sin(np.pi*(Y[:,0] + Y[:,1]))
C = C.reshape((N,1))
C = (C - np.ones((C.size,1))*min(C))/(max(C)-min(C))

print(w.shape)
print(C.shape)

#Y = np.fromfile('eady_data_5.txt',sep = " ")
#l = int(Y.size/2)
#Y = Y.reshape((l,2))
#print(Y)
#w = np.fromfile('eady_weights_5.txt',sep = " ")
#print(w.shape)
#C = np.fromfile('thetap_5.txt',sep= " ")
#C = C.reshape((l,1))
#print(C.shape)
#C = (C - np.ones((C.size,1))*min(C))/(max(C)-min(C))


img = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C,bbox,100,10000)
img.tofile("eady_laguerre_diagram.txt",sep=" ",format="%s")

[E,x,y] = pdx.periodicinx_draw_laguerre_cells_2(dens,Y,w)
#Y.tofile('points_data.txt',sep=" ",format="%s")
#x.tofile('x_data.txt',sep=" ",format="%s")
#y.tofile('y_data.txt',sep=" ",format="%s")
plt.plot(Y[:,0],Y[:,1],'.')
plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)
plt.savefig('periodic_plot.png')
