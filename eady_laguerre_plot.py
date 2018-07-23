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
rescaled = True

if rescaled:
    bbox = np.array([0., 0., 1., 1.])
    Xdens = pdx.sample_rectangle(bbox)
    f0 = np.ones(4)
    rho = np.zeros(Xdens.shape[0])
    T = ma.delaunay_2(Xdens,rho)
    dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)
    Y = np.fromfile('eady_data_5.txt',sep = " ")
    l = int(Y.size/2)
    Y = Y.reshape((l,2))
    Y[:,0] = (Y[:,0] + L)/2/L
    Y[:,1] /= H
    
else:
    bbox = np.array([-L, 0., L, H])
    Xdens = pdx.sample_rectangle(bbox)
    f0 = np.ones(4)/2/L/H
    rho = np.zeros(Xdens.shape[0])
    T = ma.delaunay_2(Xdens,rho)
    dens = pdx.Periodic_density_in_x(Xdens,f0,T,bbox)
    Y = np.fromfile('eady_data_5.txt',sep = " ")
    l = int(Y.size/2)
    Y = Y.reshape((l,2))
    

w = Y[:,0]*0.

C = np.fromfile('thetap_5.txt',sep= " ")
C = C.reshape((l,1))
C = (C - np.ones((C.size,1))*min(C))/(max(C)-min(C))
C = np.hstack((C,0.*C,1-C))

img = pdx.periodic_laguerre_diagram_to_image(dens,Y,w,C,bbox,1000,1000)
img.tofile("eady_laguerre_diagram.txt",sep=" ",format="%s")
print(img.shape)
