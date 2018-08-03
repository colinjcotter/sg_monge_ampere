import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import periodic_densities as pdx
import MongeAmpere as ma
import matplotlib.pyplot as plt
from PIL import Image

H = 1.e4
L = 1.e6

bbox = np.array([-L, 0., L, H])
Xdens = pdx.sample_rectangle(bbox)
f0 = np.ones(4)/2/L/H
rho = np.zeros(Xdens.shape[0])
T = ma.delaunay_2(Xdens,rho)
dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)

C = np.fromfile('thetap.txt',sep= " ")
l = int(C.size)
C = C.reshape((l,1))
C = (C - np.ones((C.size,1))*min(C))/(max(C)-min(C))
C = np.hstack((C,0.*C,1-C))

for n in range(380,390,10):
    print(n)
    Y = np.fromfile('points_results/Gpoints_'+str(n)+'.txt',sep = " ")
    w = np.fromfile('weights_results/weights_'+str(n)+'.txt',sep = " ")
    Y = Y.reshape((l,2))
    img = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C,bbox,1000,1000)
    img.tofile('eady_laguerre_diagram'+str(n)+'.txt',sep=" ",format="%s")
    os.remove('points_results/Gpoints_'+str(n)+'.txt')
    os.remove('weights_results/weights_'+str(n)+'.txt')
