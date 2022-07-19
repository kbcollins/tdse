#!/usr/bin/env python
# coding: utf-8

# try batching over all the p's
# test model in which v_C depends on \phi_k, \phi_{k-1}
# we'll use JAX as a GPU compiler for now
# we can switch to CuPY as long as we don't touch autograd!
# however this will involve getting rid of lax and vmap calls

from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import scipy.sparse as ss
import scipy.linalg as sl

import jax
import jax.numpy as jnp
import jax.nn
from jax import jit, lax, pmap, vmap, jacobian

import time

# set which GPU to use
devnum = 0

# number of grid points
npts = 601

# spatial extent of grid: [-L,L]
La = -80
Lb = 40

# 1d grid
x = jnp.array( np.linspace(La, Lb, npts), device=jax.devices()[devnum] )
dx = (Lb-La)/(npts-1)
print(dx)

# time step
# natural units of time
tunits = 2.4188843265857e-17
# want dt*tunits to be 2.4e-3 femtoseconds
dt = 2.4e-3*1e-15/tunits
print(dt)

# fourth-order
def laplacian1D(N, dx):
    diag=np.ones(N)
    mat=ss.spdiags([-diag,16*diag,-30*diag,16*diag,-diag],[-2,-1,0,1,2],N,N)
    return mat/(12*dx**2)

kinmat = -0.5 * laplacian1D(npts, dx).toarray()
jaxkinmat = jnp.array( kinmat, device = jax.devices()[devnum] )
evals, evecs = np.linalg.eigh(kinmat)

# kinetic propagator
jaxkinprop = jnp.array( evecs @ np.diag(np.exp(-1j*evals*dt/2)) @ evecs.conj().T,
                        device = jax.devices()[devnum] )

# external potential
Vext = -((x + 10)**2 + 1)**(-0.5)

# Wee
x1, x2 = jnp.array( np.meshgrid(x, x), device = jax.devices()[devnum] )
Wee = dx*((x1-x2)**2 + 1)**(-0.5)

# quadrature weights for Simpson's rule
w = np.ones(npts)
w[1:(npts-1):2] = 4
w[2:(npts-1):2] = 2
print(w[:4])
print(w[-4:])
w /= 3.0
w = jnp.array(w, device = jax.devices()[devnum])

# test Simpson's rule for cubic polynomial
# exact answer is 2594400
print( jnp.sum(w * (-0.25*x**3 + x**2 - x) * dx) )

# prefix for path
pathprefixes = ['./datap10/', './datap18/']
jphi0 = []
jphi1 = []
ctr = 0

for pathprefix in pathprefixes:
    # load initial Kohn-Sham state from disk
    phi0 = np.load(pathprefix+'phi0.npz')['phi0'][::2]
    jphi0.append( jnp.array(phi0, device=jax.devices()[devnum+ctr]) )
    print(phi0.shape)
    print( np.sum(w*(np.abs(phi0)**2)*dx) )

    # load initial Kohn-Sham state from disk
    phi1 = np.load(pathprefix+'phi100.npz')['phi0'][::2]
    jphi1.append( jnp.array(phi1, device=jax.devices()[devnum+ctr]) )
    print(phi1.shape)
    print( np.sum(w*(np.abs(phi1)**2)*dx) )
    ctr += 1

jallden = []
ctr = 0
for pathprefix in pathprefixes:
    allden = np.load(pathprefix+'GPUdensities.npz')['density'][::100,::2]
    print(allden.shape)
    jallden.append( jnp.array(allden, device=jax.devices()[devnum+ctr]) )
    ctr += 1

jphi0 = jnp.stack(jphi0)
jphi1 = jnp.stack(jphi1)
jallden = jnp.stack(jallden)

nlayers = 4
layerwidths = [4*npts,256,256,256,npts]
numparams = 0
for j in range(nlayers):
    numparams += layerwidths[j]*layerwidths[j+1] + layerwidths[j+1]
print(numparams)

def vcmodel(phiRk, phiRkm1, phiIk, phiIkm1, theta):
    filt = []
    si = 0
    ei = layerwidths[0]*layerwidths[1]
    filt.append( theta[si:ei].reshape((layerwidths[0],layerwidths[1])) )
    si += layerwidths[0]*layerwidths[1]
    ei += layerwidths[1]*layerwidths[2]
    filt.append( theta[si:ei].reshape((layerwidths[1],layerwidths[2])) )
    si += layerwidths[1]*layerwidths[2]
    ei += layerwidths[2]*layerwidths[3]
    filt.append( theta[si:ei].reshape((layerwidths[2],layerwidths[3])) )
    si += layerwidths[2]*layerwidths[3]
    ei += layerwidths[3]*layerwidths[4]
    filt.append( theta[si:ei].reshape((layerwidths[3],layerwidths[4])) )
    bias = []
    si += layerwidths[3]*layerwidths[4]
    ei += layerwidths[1]
    bias.append( theta[si:ei] )
    si += layerwidths[1]
    ei += layerwidths[2]
    bias.append( theta[si:ei] )
    si += layerwidths[2]
    ei += layerwidths[3]
    bias.append( theta[si:ei] )
    si += layerwidths[3]
    ei += layerwidths[4]
    bias.append( theta[si:ei] )
    inplyr = jnp.array( jnp.concatenate([phiRk, phiRkm1, phiIk, phiIkm1]), device = jax.devices()[devnum])
    h1 = jax.nn.selu( inplyr @ filt[0] + bias[0] )
    h2 = jax.nn.selu( h1 @ filt[1] + bias[1] )
    h3 = jax.nn.selu( h2 @ filt[2] + bias[2] )
    h4 = h3 @ filt[3] + bias[3]
    return h4

mydvcdthisphiR = jacobian(vcmodel, 0) # , device = jax.devices()[devnum])
mydvcdprevphiR = jacobian(vcmodel, 1) # , device = jax.devices()[devnum])
mydvcdthisphiI = jacobian(vcmodel, 2) # , device = jax.devices()[devnum])
mydvcdprevphiI = jacobian(vcmodel, 3) # , device = jax.devices()[devnum])
mydvcdtheta = jacobian(vcmodel, 4) # , device = jax.devices()[devnum])

nsteps = 300

# generate phi for this vc
def genphi(thisphi0, thisphi1, theta):
    # forward-in-time state propagation
    def bodyfun(i, phi):
        thisphiR = jnp.real(phi[i,:])
        thisphiI = jnp.imag(phi[i,:])
        n = 2*(thisphiR**2 + thisphiI**2)
        vhx = 0.5*Wee @ (w * n)
        prevphiR = jnp.real(phi[i-1,:])
        prevphiI = jnp.imag(phi[i-1,:])
        vtot = Vext + vhx + vcmodel(thisphiR, prevphiR, thisphiI, prevphiI, theta)
        jaxpotprop = jnp.diag(jnp.exp(-1j*vtot*dt))
        return phi.at[i+1].set( jaxkinprop @ (jaxpotprop @ (jaxkinprop @ phi[i,:])) )
    
    phimat = jnp.array( 
        jnp.vstack([jnp.expand_dims(thisphi0,0), jnp.expand_dims(thisphi1,0), jnp.zeros((nsteps-1, npts))]),
        device = jax.devices()[devnum])
    phipred = lax.fori_loop(1, nsteps, bodyfun, phimat)
    return phipred

jgenphi = jit(genphi, device = jax.devices()[devnum])

theta0 = jnp.array(0.01*np.random.normal(size=numparams))

# sum of squared errors cost
def gencost(thisphi0, thisphi1, thisden, theta):
    phipred = jgenphi(thisphi0, thisphi1, theta)
    denpred = 2*jnp.abs(phipred)**2
    cost = 0.5*jnp.sum(jnp.square(denpred - thisden))
    return cost

jcost = jit(gencost, device = jax.devices()[devnum])

# the real part of termR will contain jacobian of real part of propF w.r.t. thisphiR
# the imag part of termR will contain jacobian of imag part of propF w.r.t. thisphiR
# the real part of termI will contain jacobian of real part of propF w.r.t. thisphiI
# the imag part of termI will contain jacobian of imag part of propF w.r.t. thisphiI
def dFdthisphi(thisphiR, prevphiR, thisphiI, prevphiI, theta):
    n = 2.0*(thisphiR**2 + thisphiI**2)
    vhx = 0.5*Wee @ (w * n)
    vtot = Vext + vhx + vcmodel(thisphiR, prevphiR, thisphiI, prevphiI, theta)
    jaxpotprop = jnp.exp(-1j*vtot*dt)
    dvdtR = mydvcdthisphiR(thisphiR, prevphiR, thisphiI, prevphiI, theta)
    dvdtI = mydvcdthisphiI(thisphiR, prevphiR, thisphiI, prevphiI, theta)
    dVdtphiR = 2*Wee * (w * thisphiR) + dvdtR
    dVdtphiI = 2*Wee * (w * thisphiI) + dvdtI
    thisphi = thisphiR + 1j*thisphiI
    term1R = (-1j*dt)*jnp.einsum('lq,q,qm,qr,r->lm',jaxkinprop,jaxpotprop,dVdtphiR,jaxkinprop,thisphi)
    term1I = (-1j*dt)*jnp.einsum('lq,q,qm,qr,r->lm',jaxkinprop,jaxpotprop,dVdtphiI,jaxkinprop,thisphi)
    term2 = jaxkinprop @ jnp.diag(jaxpotprop) @ jaxkinprop
    termR = jnp.array( term1R + term2, device = jax.devices()[devnum])
    termI = jnp.array( term1I + 1j*term2,  device = jax.devices()[devnum])
    return termR, termI

# the real part of termR will contain jacobian of real part of propF w.r.t. prevphiR
# the imag part of termR will contain jacobian of imag part of propF w.r.t. prevphiR
# the real part of termI will contain jacobian of real part of propF w.r.t. prevphiI
# the imag part of termI will contain jacobian of imag part of propF w.r.t. prevphiI
def dFdprevphi(thisphiR, prevphiR, thisphiI, prevphiI, theta):
    n = 2.0*(thisphiR**2 + thisphiI**2)
    vhx = 0.5*Wee @ (w * n)
    vtot = Vext + vhx + vcmodel(thisphiR, prevphiR, thisphiI, prevphiI, theta)
    jaxpotprop = jnp.exp(-1j*vtot*dt)
    dvdpR = mydvcdprevphiR(thisphiR, prevphiR, thisphiI, prevphiI, theta)
    dvdpI = mydvcdprevphiI(thisphiR, prevphiR, thisphiI, prevphiI, theta)
    thisphi = thisphiR + 1j*thisphiI
    termR = jnp.array(
        (-1j*dt)*jnp.einsum('lq,q,qm,qr,r->lm',jaxkinprop,jaxpotprop,dvdpR,jaxkinprop,thisphi),
        device = jax.devices()[devnum])
    termI = jnp.array(
        (-1j*dt)*jnp.einsum('lq,q,qm,qr,r->lm',jaxkinprop,jaxpotprop,dvdpI,jaxkinprop,thisphi),
        device = jax.devices()[devnum])
    return termR, termI

def dFdtheta(thisphiR, prevphiR, thisphiI, prevphiI, theta):
    n = 2.0*(thisphiR**2 + thisphiI**2)
    vhx = 0.5*Wee @ (w * n)
    vtot = Vext + vhx + vcmodel(thisphiR, prevphiR, thisphiI, prevphiI, theta)
    jaxpotprop = jnp.exp(-1j*vtot*dt)
    dVdtheta = mydvcdtheta(thisphiR, prevphiR, thisphiI, prevphiI, theta)
    thisphi = thisphiR + 1j*thisphiI
    term = jnp.array(
        (-1j*dt)*jnp.einsum('lq,q,qm,qr,r->lm',jaxkinprop,jaxpotprop,dVdtheta,jaxkinprop,thisphi),
        device = jax.devices()[devnum])
    return term

def adjprop(trueden, phipred, theta):
    denpred = 2*(jnp.real(phipred)**2 + jnp.imag(phipred)**2)
    
    def bodylamb(j, lamb):
        term1R = 4*((denpred[nsteps-j-1,:] - trueden[nsteps-j-1,:]) * jnp.real(phipred[nsteps-j-1,:]))
        term1I = 4*((denpred[nsteps-j-1,:] - trueden[nsteps-j-1,:]) * jnp.imag(phipred[nsteps-j-1,:]))
        phiRj = jnp.real(phipred[nsteps-j-1,:])
        phiIj = jnp.imag(phipred[nsteps-j-1,:])
        phiRjm1 = jnp.real(phipred[nsteps-j-2,:])
        phiIjm1 = jnp.imag(phipred[nsteps-j-2,:])
        phiRjp1 = jnp.real(phipred[nsteps-j,:])
        phiIjp1 = jnp.imag(phipred[nsteps-j,:])
        dR, dI = dFdthisphi(phiRj, phiRjm1, phiIj, phiIjm1, theta)
        term2R = lamb[nsteps-j,:] @ jnp.vstack([jnp.real(dR), jnp.imag(dR)])
        term2I = lamb[nsteps-j,:] @ jnp.vstack([jnp.real(dI), jnp.imag(dI)])
        dRprev, dIprev = dFdprevphi(phiRjp1, phiRj, phiIjp1, phiIj, theta)
        term2Rfut = lamb[nsteps-j+1,:] @ jnp.vstack([jnp.real(dRprev), jnp.imag(dRprev)])
        term2Ifut = lamb[nsteps-j+1,:] @ jnp.vstack([jnp.real(dIprev), jnp.imag(dIprev)])
        prevlamb = jnp.hstack([term1R + term2R + term2Rfut, term1I + term2I + term2Ifut])
        return lamb.at[nsteps-1-j].set( prevlamb )
    
    flambR = 4*((denpred[nsteps,:] - trueden[nsteps,:]) * jnp.real(phipred[nsteps,:]))
    flambI = 4*((denpred[nsteps,:] - trueden[nsteps,:]) * jnp.imag(phipred[nsteps,:]))
    aflambR = 4*((denpred[nsteps-1,:] - trueden[nsteps-1,:]) * jnp.real(phipred[nsteps-1,:]))
    aflambI = 4*((denpred[nsteps-1,:] - trueden[nsteps-1,:]) * jnp.imag(phipred[nsteps-1,:]))
    dRnm1, dInm1 = dFdthisphi(jnp.real(phipred[nsteps-1,:]),
                              jnp.real(phipred[nsteps-2,:]), 
                              jnp.imag(phipred[nsteps-1,:]),
                              jnp.imag(phipred[nsteps-2,:]), theta)
    aflambR += flambR @ jnp.real(dRnm1) + flambI @ jnp.imag(dRnm1)
    aflambI += flambR @ jnp.real(dInm1) + flambI @ jnp.imag(dInm1)
    lambmat = lax.stop_gradient(jnp.array(
        jnp.vstack([jnp.zeros((nsteps-1, 2*npts)), 
        jnp.hstack([aflambR, aflambI]),
        jnp.hstack([flambR, flambI])]),
        device = jax.devices()[devnum]))
    lambpred = lax.fori_loop(1, nsteps-2, bodylamb, lambmat)
    return lambpred

jadjprop = jit(adjprop, device = jax.devices()[devnum])

def adjgrad(thisphi0, thisphi1, thisden, theta):
    # forward-in-time state propagation
    phipred = genphi(thisphi0, thisphi1, lax.stop_gradient(theta))
    denpred = 2*(jnp.real(phipred)**2 + jnp.imag(phipred)**2)
    
    # backward-in-time adjoint propagation
    lambpred = adjprop(thisden, lax.stop_gradient(phipred), lax.stop_gradient(theta))

    # compute the gradient with respect to vC
    def bodyatg(j, atg):
        thisphiR = jnp.real(phipred[j,:])
        prevphiR = jnp.real(phipred[j-1,:])
        thisphiI = jnp.imag(phipred[j,:])
        prevphiI = jnp.imag(phipred[j-1,:])
        myderiv = dFdtheta(thisphiR, prevphiR, thisphiI, prevphiI, theta)
        return atg + lambpred[j+1,:npts] @ jnp.real(myderiv) + lambpred[j+1,npts:] @ jnp.imag(myderiv)
    
    atgpred = lax.fori_loop(1, nsteps, bodyatg, 
                            lax.stop_gradient(jnp.array(jnp.zeros(numparams), device = jax.devices()[devnum])))   
    return atgpred

jadjgrad = jit(adjgrad) # , device = jax.devices()[devnum])

start = time.time()
myres = jadjgrad(jphi0[0], jphi1[0], jallden[0], theta0)
end = time.time()
print(end-start)

jvcost = pmap(jcost, in_axes=(0,0,0,None))

start = time.time()
oldcost = jvcost(jphi0, jphi1, jallden, theta0)
end = time.time()
print(end-start)

jvadjgrad = pmap(jadjgrad, in_axes=(0,0,0,None))

start = time.time()
test = jvadjgrad(jphi0, jphi1, jallden, theta0)
end = time.time()
print(end-start)

print(oldcost)
print(theta0.shape)
print(numparams)

def siobj(x):
    jx = jnp.array(x, device=jax.devices()[0])
    return np.mean( np.asarray( jvcost( jphi0, jphi1, jallden, jx ) ), axis=0 )

def sigrad(x):
    jx = jnp.array(x, device=jax.devices()[0])
    thisgrad = jvadjgrad( jphi0, jphi1, jallden, jx ) 
    return np.mean( np.asarray( thisgrad ), axis=0 )

import scipy.optimize as so

def mycb(xk):
    np.savez('Dec5thetaTRAINfullphiTWO.npz',bfgsvc=xk)
    return False

res = so.minimize( siobj, 
                   x0 = theta0,
                   method = 'L-BFGS-B', 
                   jac = sigrad,
                   callback = mycb,
                   options = {'iprint': 1, 'ftol': 1e-10, 'gtol': 1e-10} )

np.savez('Dec5thetaTRAINfullphiTWOfinal.npz',bfgsvc=res.x)



