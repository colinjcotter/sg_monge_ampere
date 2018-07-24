import matplotlib
matplotlib.use('Agg')

import numpy as np
import periodic_densities as pdx
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from eady_initial import initialise_points, eady_OT
from PIL import Image

N = 50
H = 1.e4
L = 1.e6

colour = False

bbox = np.array([-L, 0., L, H])
Xdens = pdx.sample_rectangle(bbox)
f0 = np.ones(4)/2/L/H
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)

Y = np.fromfile('Gpoints_336.txt',sep = " ")
w = np.fromfile('weights_336.txt',sep=" ")
C = np.fromfile('thetap.txt',sep= " ")

l = int(Y.size/2)
Y = Y.reshape((l,2))
C = C.reshape((l,1))
C = (C - np.ones((C.size,1))*min(C))/(max(C)-min(C))

if colour:
    C = np.hstack((C,0.*C,1-C))
    img = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C,bbox,1000,1000)
    img.tofile("eady_laguerre_diagram.txt",sep=" ",format="%s")

else:
    img = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C,bbox,1000,1000)
    img.tofile("eady_laguerre_diagram.txt",sep=" ",format="%s")

