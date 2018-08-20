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

Y = np.fromfile('Gpoints.txt',sep = " ")
w = np.fromfile('weights.txt',sep=" ")
C1 = np.fromfile('thetap_init.txt',sep= " ")
C2 = np.fromfile('thetap_final.txt',sep= " ")

l = int(Y.size/2)
Y = Y.reshape((l,2))
C1 = C1.reshape((l,1))
C1 = (C1 - np.ones((C1.size,1))*min(C1))/(max(C1)-min(C1))
C2 = C2.reshape((l,1))
C2 = (C2 - np.ones((C2.size,1))*min(C2))/(max(C2)-min(C2))

if colour:
    C1 = np.hstack((C1,0.*C1,1-C1))
    img1 = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C1,bbox,1000,1000)
    img1.tofile("laguerre_diagram_init.txt",sep=" ",format="%s")
    C2 = np.hstack((C2,0.*C2,1-C2))
    img2 = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C2,bbox,1000,1000)
    img2.tofile("laguerre_diagram_final.txt",sep=" ",format="%s")
    
else:
    img1 = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C1,bbox,1000,1000)
    img1.tofile("laguerre_diagram_init.txt",sep=" ",format="%s")
    img2 = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C2,bbox,1000,1000)
    img2.tofile("laguerre_diagram_final.txt",sep=" ",format="%s")
    

