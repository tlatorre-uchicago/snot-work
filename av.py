from __future__ import division
import random
import numpy as np

def norm(x):
    "Returns the norm of the vector `x`."
    return np.sqrt((x*x).sum(-1))

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

def bar(r):
    from scipy.special import expn
    x1 = (R-r)/L
    x2 = (r+R)/L
    return -expn(1,x1) + expn(1,x2)

def baz(r):
    from scipy.special import expn
    x1 = (R-r)/L
    x2 = (r+R)/L
    return r*(-expn(1,x1) + expn(1,x2))

# scattering length
L = 26.3e-2
# sphere radius
R = 6
# number of monte carlo events
N = 10000000

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # pick random points on a sphere of radius R, draw an isotropic direction
    # and move the points a distance drawn from an exponential with mean L
    x = uniform_sphere(N)*R + np.random.exponential(L,size=N)[:,np.newaxis]*uniform_sphere(N)

    # get the distance from the center
    r = norm(x)

    bins = np.linspace(5,6,1000)
    dx = bins[1] - bins[0]
    bincenters = (bins[1:] + bins[:-1])/2
    hist, _ = np.histogram(r,bins)

    y1 = bar(bincenters)
    y1 /= y1.sum()*dx
    y2 = baz(bincenters)
    y2 /= y2.sum()*dx

    plt.hist(r, bins=bins, normed=True)
    plt.plot(bincenters,y1,lw=2)
    plt.plot(bincenters,y2,lw=2)

    plt.show()
