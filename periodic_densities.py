import matplotlib
matplotlib.use('Agg')

import MongeAmpere as ma
import numpy as np
import scipy as sp
import pylab
import pdb
import matplotlib.pyplot as plt
import matplotlib.tri as tri

class Periodic_density_in_x (ma.ma.Density_2):
    def __init__(self, X, f, T, bbox): 
        self.x0 = np.array([bbox[0],bbox[1]]);
        self.x1 = np.array([bbox[2],bbox[3]]);
        self.u = self.x1 - self.x0;
        ma.ma.Density_2.__init__(self, X,f,T)

    def to_fundamental_domain(self,Y):
        N = Y.shape[0];
        Y[:,0] = (Y[:,0] - np.tile(self.x0,(N,1))[:,0]) / np.tile(self.u,(N,1))[:,0]; 
        Y[:,0] = Y[:,0] - np.floor(Y[:,0]);
        Y[:,0] = np.tile(self.x0,(N,1))[:,0] + Y[:,0] * np.tile(self.u,(N,1))[:,0];
        return Y;

    # FIXME
    def kantorovich(self,Y,nu,w):
        N = len(nu);
        # create copies of the points, so as to cover the neighborhood
        # of the fundamental domain.
        Y0 = self.to_fundamental_domain(Y)
        x = self.u[0]
        y = self.u[1]
        v = np.array([[0,0], [x,0], [-x,0]]);
        Yf = np.zeros((3*N,2))
        wf = np.hstack((w,w,w));
        for i in xrange(0,3):
            Nb = N*i; Ne = N*(i+1)
            Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

        # sum the masses of each "piece" of the Voronoi cells
        #segmentation fault for regular mesh
        #pdb.set_trace()
        [f,mf,hf] = ma.ma.kantorovich_2(self, Yf, wf);

        m = np.zeros(N);
        for i in xrange(0,3):
            Nb = N*i; Ne = N*(i+1);
            m += mf[Nb:Ne]

        # adapt the Hessian by correcting indices of points. we use
        # the property that elements that appear multiple times in a
        # sparse matrix are summed
        h = (hf[0], (np.mod(hf[1][0], N), np.mod(hf[1][1], N)))

        # remove the linear part of the function
        f = f - np.dot(w,nu);
        g = m - nu;
        H = sp.sparse.csr_matrix(h,shape=(N,N))
        return f,m,g,H;

    def lloyd(self,Y,w=None):
        if w is None:
            w = np.zeros(Y.shape[0]);
        N = Y.shape[0];
        Y0 = self.to_fundamental_domain(Y)

        # create copies of the points, so as to cover the neighborhood
        # of the fundamental domain.
        x = self.u[0]
        y = self.u[1]
        v = np.array([[0,0], [x,0], [-x,0]]);
        Yf = np.zeros((3*N,2))
        wf = np.hstack((w,w,w));
        for i in xrange(0,3):
            Nb = N*i; Ne = N*(i+1)
            Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

        # sum the moments and masses of each "piece" of the Voronoi
        # cells
        [mf,Yf,If] = ma.ma.moments_2(self, Yf, wf);

        Y = np.zeros((N,2));
        m = np.zeros(N);
        for i in xrange(0,3):
            Nb = N*i; Ne = N*(i+1);
            m += mf[Nb:Ne]
            ww = np.tile(mf[Nb:Ne],(2,1)).T
            Y += Yf[Nb:Ne,:] - ww * np.tile(v[i,:],(N,1))

        # rescale the moments to get centroids
        Y /= np.tile(m,(2,1)).T
        Y = self.to_fundamental_domain(Y);
        return (Y,m)

    def moments(self,Y,w=None):
        if w is None:
            w = np.zeros(Y.shape[0]);
        N = Y.shape[0];
        Y0 = self.to_fundamental_domain(Y)

        # create copies of the points, so as to cover the neighborhood
        # of the fundamental domain.
        x = self.u[0]
        y = self.u[1]
        v = np.array([[0,0], [x,0], [-x,0]]);
        Yf = np.zeros((3*N,2))
        wf = np.hstack((w,w,w));
        for i in xrange(0,3):
            Nb = N*i; Ne = N*(i+1)
            Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

        # sum the moments and masses of each "piece" of the Voronoi
        # cells
        [mf,Yf,If] = ma.ma.moments_2(self, Yf, wf)
        return mf,Yf,If

    #FIX
    def averages(self,Y,w=None):
        if w is None:
            w = np.zeros(Y.shape[0]);
        N = Y.shape[0];
        Y0 = self.to_fundamental_domain(Y)

        # create copies of the points, so as to cover the neighborhood
        # of the fundamental domain.
        x = self.u[0]
        y = self.u[1]
        v = np.array([[0,0], [x,0], [-x,0]]);
        Yf = np.zeros((3*N,2))
        wf = np.hstack((w,w,w));
        for i in xrange(0,3):
            Nb = N*i; Ne = N*(i+1)
            Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

        # sum the moments and masses of each "piece" of the Voronoi
        # cells
        [mf,Yf0,If] = ma.ma.moments_2(self, Yf, wf)

        Ya = np.zeros((N,2)); #average of X-x, z 
        m = np.zeros(N); #total area of cell
        for i in xrange(0,3):
            Nb = N*i; Ne = N*(i+1);
            m += mf[Nb:Ne]
            ww = np.tile(mf[Nb:Ne],(2,1)).T
            Ya[:,1] += Yf[Nb:Ne,1] - ww * np.tile(v[i,1],(N,1))
            Ya[:,0] += Yf0[Nb:Ne,0] - Yf[Nb:Ne,0]*mf[Nb:Ne,0]

        Ya /= np.tile(m,(2,1)).T
        Ya = self.to_fundamental_domain(Y);
        return Ya 

# generate density
def sample_rectangle(bbox):
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]
    x = [x0, x1, x1, x0]
    y = [y0, y0, y1, y1]
    X = np.vstack((x,y)).T
    return X

def periodicinx_draw_laguerre_cells_2(pxdens,Y,w):
    # draw laguerre cells when boundary is periodic in
    # x direction
    N = Y.shape[0]
    Y0 = pxdens.to_fundamental_domain(Y)
    x0 = pxdens.u[0]
    v = np.array([[0,0], [x0,0], [-x0,0]])
    Yf = np.zeros((3*N,2))
    wf = np.hstack((w,w,w))
    for i in xrange(0,3):
        Nb = N*i
        Ne = N*(i+1)
        Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))
    E = pxdens.restricted_laguerre_edges(Yf,wf)
    print(E)
    nan = float('nan')
    N = E.shape[0]
    x = np.zeros(3*N)
    y = np.zeros(3*N)
    a = np.array(range(0,N))
    x[3*a] = E[:,0]
    x[3*a+1] = E[:,2]
    x[3*a+2] = nan
    y[3*a] = E[:,1]
    y[3*a+1] = E[:,3]
    y[3*a+2] = nan
    #print(x)
    #print(y)
    return E,x,y
    #plt.plot(Y[:,0],Y[:,1],'.')
    #plt.plot(x,y,color=[1,0,0],linewidth=1,aa=True)
    #plt.savefig('periodic_plot.png')
