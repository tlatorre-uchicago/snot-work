import numpy as np
from scipy.optimize import fmin

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

def intersect_sphere(pos, dir, r=1):
    d1 = -(dir*pos).sum(axis=1) + np.sqrt((dir*pos).sum(axis=1)**2 - (pos**2).sum(axis=1) + r**2)
    d2 = -(dir*pos).sum(axis=1) - np.sqrt((dir*pos).sum(axis=1)**2 - (pos**2).sum(axis=1) + r**2)
    d = np.max(np.vstack((d1,d2)), axis=0)
    return pos + d[:,np.newaxis]*dir

def norm(x):
    "Returns the norm of the vector `x`."
    return np.sqrt((x*x).sum(-1))

TAU = 1

xtrue = 0

def foo(pars, ti):
    x, = pars
    if (ti < x).any():
        return 1e9
    return (ti-x).sum()/TAU

N = [1,10,100]#range(10,1000,10)
J = 1000

results = np.empty((len(N),J), dtype=float)
for i, n in enumerate(N):
    pos = xtrue

    for j in range(J):
        d = 10.0
	ti = np.random.exponential(TAU,n)
	ti += d

	fopt = 1e9
	while fopt > 1e8:
            x0 = np.random.random(1)
	    xopt, fopt, iter, funcalls, warnflag = fmin(foo,x0, args=(ti,), disp=True, full_output=True)
        results[i][j] = xopt

std = np.std(results,axis=1)
mean = np.mean(results,axis=1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.plot(N,std[:,0])
    plt.plot(N,std[:,1])
    plt.plot(N,std[:,2])
    plt.plot(N,10.0/np.array(N))
    plt.plot(N,1.0/np.sqrt(np.array(N)))
    plt.show()
