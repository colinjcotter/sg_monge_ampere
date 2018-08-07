import matplotlib
matplotlib.use('Agg')

import os, sys
parentpath = os.path.abspath("..")
if parentpath not in sys.path:
    sys.path.insert(0, parentpath)
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import periodic_densities as pdx
from PIL import Image

H = 1.e4
L = 1.e6

colour = True

bbox = np.array([-L, 0., L, H])
Xdens = pdx.sample_rectangle(bbox)
f0 = np.ones(4)/2/L/H
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)

Y = np.fromfile('Gpoints_1728000.txt',sep = " ")
w = np.fromfile('weights_1728000.txt',sep=" ")
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

