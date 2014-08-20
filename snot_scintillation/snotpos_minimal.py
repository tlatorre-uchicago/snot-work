from __future__ import division
import numpy as np
from scipy.optimize import fmin
from scipy.special import erfc, expi

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
    """
    Returns the intersection between rays starting at `pos` and traveling
    in the direction `dir` and a sphere of radius `r`.
    """
    d1 = -(dir*pos).sum(axis=1) + np.sqrt((dir*pos).sum(axis=1)**2 - \
        (pos**2).sum(axis=1) + r**2)
    d2 = -(dir*pos).sum(axis=1) - np.sqrt((dir*pos).sum(axis=1)**2 - \
        (pos**2).sum(axis=1) + r**2)
    d = np.max(np.vstack((d1,d2)), axis=0)
    return pos + d[:,np.newaxis]*dir

def norm(x):
    "Returns the norm of the vector `x`."
    return np.sqrt((x*x).sum(-1))

class ScintillationProfile(object):
    def __init__(self, r=12000.0, tau=4.0, c=299.8):
        self.r = r
        self.tau = tau
        self.c = c

    def pdf(self, t, l):
        """
        Returns the pdf for a hit at time `t` from an event
        at radius `l`.
        """
        r, c, tau = self.r, self.c, self.tau

        t = np.atleast_1d(t)

        tcut = (l+r)/c
        tmin = (r-l)/c

        x = t.copy()
        x[t > tcut] = tcut

        y = c*x*tau*np.exp(tmin/tau)*(r+l-c*tau)
        y += np.exp(x/tau)*(l**2-r**2+c**2*x*tau)*tau
        y += x*(l**2-r**2)*(expi(tmin/tau) - expi(x/tau))
        y /= 4*c*l*x*tau
        y *= np.exp(-t/tau)/tau
        y[t < tmin] = 0

        return y if y.size > 1 else y.item()

    def cdf(self, t, l):
        """
        Returns the cdf for a hit at time `t` from an event
        at radius `l`.
        """
        def _cdf(t):
            x = np.linspace((self.r-l)/self.c,t,100)
            return np.trapz(self.pdf(x,l),x)

        if np.iterable(t):
            return np.array([_cdf(x) for x in t])
        else:
            return _cdf(t)

    def gen_times(self, l=0, n=100):
        """
        Generates a set of `n` hit times for an event at radius
        `l`.
        """
        pos = np.atleast_2d((l,0,0))
        dir = uniform_sphere(n)
        hit = intersect_sphere(pos,dir,r=self.r)
        d = norm(hit-pos)
        return np.random.exponential(self.tau,n) + d/self.c

    def fit(self, t, n=10, retry=True):
        """
        Fit for the radius of an event from a given set of
        hit times `t`. Seed the initial position by searching
        for the most likely radius at `n` radii. If the fit
        fails and `retry` is True, try the fit again by seeding
        the fit with the best point from `n`*10 trial radii.
        """
        def nll(pars):
            l, = pars
            return -np.log(self.pdf(t,l=l)).sum()

        # seed the fit with the best radius in l0
        l0 = np.linspace(0,self.r,n)
        x0 = [-np.log(self.pdf(t,x)).sum() for x in l0]
        x0 = l0[np.nanargmin(x0)]

        xopt, fopt, iter, funcalls, warnflag = \
            fmin(nll,[x0],disp=False,full_output=True)

        if warnflag:
            if n < 1e3 and retry:
                print 'retrying with n=%i' % (n*10)
                return self.fit(t,n*10)
            print 'fit failed.'
        return xopt

    def ks_test(self, t, l):
        """
        Returns the KS test statistic for the hit times `t` at
        radius `l`.
        """
        ti = np.linspace(0,(self.r+l)/self.c + 2*self.tau, 100)
        return np.max(np.abs((t < ti[:,np.newaxis]).sum(1)/len(t) - self.cdf(ti,l)))
