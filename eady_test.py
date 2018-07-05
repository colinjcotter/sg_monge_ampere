import matplotlib
matplotlib.use('Agg')

import numpy as np
from periodic_densities import Periodic_density_in_x, sample_rectangle
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from eady_initial import initialise_points, eady_OT, forward_euler_sg

timestep = True

N = 60
H = 1.e4
L = 1.e6

bbox = np.array([-L, 0., L, H])
Xdens = sample_rectangle(bbox)
f0 = np.ones(4)/(H*2*L)
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = Periodic_density_in_x(Xdens,f0,T,bbox)

[Y, thetap] = initialise_points(N, bbox, RegularMesh = False)
print(thetap.shape)
thetap.tofile('thetap.txt',sep=" ",format="%s")

if not timestep:
    w = eady_OT(Y, bbox, dens, verbose = True)
    [Y, m] = dens.lloyd(Y,w)
    print(Y.shape)
    Y.tofile('eady_data.txt',sep=" ",format="%s")
    sc = plt.scatter(Y[:,0],Y[:,1],c=thetap,cmap="plasma")
    plt.colorbar(sc)
    plt.savefig('final.png')
    #plt.show()

else:
    tf = 60*60*24*6
    Y = forward_euler_sg(Y, dens, tf, bbox)
    #print(Y[:,0].min())
    #print(Y[:,0].max())
    print(Y.shape)
    Y.tofile('eady_data.txt',sep=" ",format="%s")
    #sc = plt.scatter(Y[:,0],Y[:,1],c=thetap,cmap="plasma")
    #plt.colorbar(sc)
    #plt.savefig('eady_image.png')
    #plt.show()
