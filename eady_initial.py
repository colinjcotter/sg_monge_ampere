import numpy as np
npts = 100
dx = 1./npts
s = np.arange(dx*0.5,1.,dx)
s1 = np.arange(-1. + dx*0.5,1.,dx)
H = 1.e4
L = 1.e6
x,z = np.meshgrid(s1*L,s*H)
nx = x.size
x = np.reshape(x,(nx,))
z = np.reshape(z,(nx,))
Y = np.array([x,z]).T

Nsq = 2.5e-5
g = 10.
f = 1.e-4
theta0 = 300
C = 3e-6
B = 1.0e-3* Nsq * theta0 * H / g

thetap = Nsq*theta0*z/g + B*np.sin(np.pi*(x/L + z/H))
vg = B*g*H/L/f*np.sin(np.pi*(x/L + z/H)) - 2*B*g*H/np.pi/L/f*np.cos(np.pi*x/L)

X = vg/f + x
Z = g*thetap/f/f/theta0

from matplotlib import pyplot
pyplot.plot(X,Z,'.')

pyplot.show()
