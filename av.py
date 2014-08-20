from __future__ import division
import random
import numpy as np
import math

def norm(x):
    "Returns the norm of the vector `x`."
    return np.sqrt((x*x).sum(-1))

def normalize(x):
    "Returns unit vectors in the direction of `x`."
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return (x/norm(x)[:,np.newaxis]).squeeze()
def uniform_sphere(size=None, dtype=np.double):
    """
    Generate random points isotropically distributed across the unit sphere.

    Args:
        - size: int, *optional*
            Number of points to generate. If no size is specified, a single
            point is returned.

    Source: Weisstein, Eric W. "Sphere Point Picking." Mathworld. 
    """

    theta, u = np.random.uniform(0.0, 2*np.pi, size), \
        np.random.uniform(-1.0, 1.0, size)

    c = np.sqrt(1-u**2)

    if size is None:
        return np.array([c*np.cos(theta), c*np.sin(theta), u])

    points = np.empty((size, 3), dtype)

    points[:,0] = c*np.cos(theta)
    points[:,1] = c*np.sin(theta)
    points[:,2] = u

    return points

def foo(r, l, x, c):
    return c*r**x*np.exp(-(R-r)/l)

def bar(r):
    from scipy.special import expn
    x1 = (R-r)/L
    x2 = (r+R)/L
    return -expn(1,x1) + expn(1,x2)

def fit_gauss(hist,bins):
    from scipy.optimize import fmin
    bincenters = (bins[1:] + bins[:-1])/2
    #hist, _ = np.histogram(x,bins)
    hist_sigma = hist.copy()
    hist_sigma[hist == 0] = 1
    def foo(args):
        l, x, c = args
        pdf = c*bincenters**np.abs(x)*np.exp(-(R-bincenters)/l)#norm.pdf(bincenters,mu,std)
        return np.sum((pdf-hist)**2/hist_sigma)

    return fmin(foo,[0.1,0,1000],maxfun=1e6)

L = 1
R = 100
N = 10000000
x = uniform_sphere(N)*R + np.random.exponential(L,size=N)[:,np.newaxis]*uniform_sphere(N)

r = norm(x)

import matplotlib.pyplot as plt

bins = np.linspace(90,100,100)
dx = bins[1] - bins[0]
bincenters = (bins[1:] + bins[:-1])/2
hist, _ = np.histogram(r,bins)
#np.savetxt('hist.txt',np.dstack((100-bincenters[::-1],hist[::-1])).squeeze())
#result = fit_gauss(hist,bins)
#print result
y = bar(bincenters)#foo(bincenters,*result)
y /= y.sum()*dx
plt.hist(r, bins=bins, normed=True)
plt.plot(bincenters,y,lw=2)
plt.show()
