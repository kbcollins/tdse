from jax.config import config
config.update("jax_enable_x64", True)

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax.numpy as jnp
from jax import grad, jit, jacobian, vmap, lax

import scipy.linalg as sl
import scipy.integrate as si
import scipy.optimize as so
import time

import numpy as np

import linearbases 
import learnschro as ls

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# our Fourier representation will use basis functions from n = -nmax to n = nmax
nmax = 8

# size of spatial domain
biga = np.pi

# true potential
def v(x):
    return 0.5*x**2

# rounded box function initial condition
def psi0(x):
    return (1.0 + np.tanh((1 - x**2)/0.5))/2.58046

# let's take our new class for a spin
# we always need a Fourier model to do the forward simulation and/or
# to represent the problem spectrally
fourmodel = linearbases.fourier(biga, 1025, nmax)

# UNCOMMENT ONE OF THE FOLLOWING TWO LINES
# IF YOU WANT THE TRUE POTENTIAL TO BE REPRESENTED USING CHEBYSHEV or FOURIER
truemodel = linearbases.chebyshev(biga, 1025, nmax, 5)
# truemodel = fourmodel

# now ask the true model to represent our potential v(x)
truemodel.represent(v)

# use the class again to represent the initial condition!
initcond = linearbases.fourier(biga, 1025, nmax)
ainit = initcond.represent(psi0)

# set up a learnschro object purely for forward sim
dt = 0.001
nsteps = 200
fs = ls.learnschro(biga, dt, nmax, nsteps=nsteps, ic=ainit, kinmat=fourmodel.kmat())

# actually use learnschro to do the forward sim
_, amat = fs.jobjtraj(truemodel.vmat())

# set up a learnschro object for actual learning
ip = ls.learnschro(biga, dt, nmax, obj='WF', ic=ainit, wfdata=amat, kinmat=fourmodel.kmat())

# generate a random theta
repsize = 4
thetarand = jnp.array(np.random.normal(size=repsize+1))

# set up a model for learning
adjmodel = linearbases.chebyshev(biga, 1025, nmax, repsize, theta=thetarand)

# wrapper for objective
def J(theta, model):
    model.settheta(theta)
    jvhatmat = jnp.array(model.vmat())
    obj, _ = ip.jobjtraj(jvhatmat)
    return obj

# wrapper for gradient
def gradJ(theta, model):
    model.settheta(theta)
    jvhatmat = jnp.array(model.vmat())
    jctrmats = jnp.array(model.grad())
    _, grad, _ = ip.jadjgrad(jvhatmat, jctrmats)
    return grad

# wrapper for hessian
def hessJ(theta, model):
    model.settheta(theta)
    jvhatmat = jnp.array(model.vmat())
    jctrmats = jnp.array(model.grad())
    _, _, _, hess, _ = ip.jadjhess(jvhatmat, jctrmats)
    return hess

# use scipy.optimize.minimize
res = so.minimize(fun=J, jac=gradJ, hess=hessJ, x0=thetarand, 
        method='trust-constr', options={'verbose': 2}, args=(adjmodel, ))



