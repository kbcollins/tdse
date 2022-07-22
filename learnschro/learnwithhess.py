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

# our Fourier representation will use basis functions from n = -nmax to n = nmax
nmax = 16

# size of spatial domain
a = np.pi

def v(x):
    return 0.5*x**2

# some sampling and FFT tricks
# this integer '100' can be adjusted to improve resolution
# if we set f_sample = 2*nmax, then we are sampling at the Nyquist frequency
f_sample = 100 * 2 * nmax
t, dt = np.linspace(-a, a, f_sample+2, endpoint=False, retstep=True)
y = (np.fft.rfft(v(t)) / t.size)[:(2*nmax+1)]

# this stores the Fourier series coefficients for n-m=0 to n-m=2*nmax
vrow = y * (-1)**np.arange(2*nmax+1)

# create Toeplitz matrix
vmat = sl.toeplitz(r=vrow,c=np.conj(vrow))

# kinetic matrix
kmat = np.diag( np.arange(-nmax,nmax+1)**2 * np.pi**2 / (2*a**2) )

# Hamiltonian matrix 
hmat = kmat + vmat

# check whether Hamiltonian is Hermitian
print(np.linalg.norm( hmat - np.conj(hmat.T) ))

# eigendecomposition
spec, states = np.linalg.eigh(hmat)

# check whether this diagonalizes hmat
# note that if this is close to zero it means that
# hmat = states @ np.diag(spec) @ np.conj(states).T
np.linalg.norm( hmat @ states - states @ np.diag(spec) )

# check whether we have a unitary matrix
print(np.linalg.norm( states @ np.conj(states).T - np.eye(2*nmax+1) ))

np.linalg.norm( states @ np.diag(spec) @ states.conj().T - hmat )

# find indices that sort eigenvalues
ordering = np.argsort(spec)

# spatial grid for the purposes of plotting
xvec = np.linspace(-a, a, 1025)

# convert basis coefficients into wavefunction on grid by matrix multiplication
nvec = np.arange(-nmax,nmax+1)
convmat = np.exp(1j*np.pi*np.outer(nvec, xvec)/a)/np.sqrt(2*a)

# pick out and plot ground state
groundstate = ordering[1]
wavefn = states[:,groundstate] @ convmat
# plt.plot(xvec, -np.real(wavefn))
# plt.show()

# check normalization
print(np.sum(np.abs(wavefn)**2 * (xvec[1]-xvec[0])))

# rounded box function
def psi0(x):
    return (1.0 + np.tanh((1 - x**2)/0.5))/2.58046

vraw = np.zeros(nmax+1, dtype=np.complex128)
for thisn in range(nmax+1):
    def integ(x):
        return (2*a)**(-0.5)*np.exp(-1j*np.pi*thisn*x/a)*psi0(x)
    def rinteg(x):
        return np.real(integ(x))
    def iinteg(x):
        return np.imag(integ(x))
    vraw[thisn] = si.quad(rinteg, a=-a, b=a)[0] + 1j*si.quad(iinteg, a=-a, b=a)[0]

ainit = np.concatenate([np.conjugate(np.flipud(vraw[1:])), vraw])

# # set the time step and compute the propagator matrix
# # note that we are reusing the spec, states eigendecomposition of hmat computed above
dt = 0.001
prop = states @ np.diag(np.exp(-1j*spec*dt)) @ np.conj(states.T)

# propagate the "a" vector of coefficients as defined above
nsteps = 200
amat = np.zeros((nsteps+1, 2*nmax+1), dtype=np.complex128)
amat[0,:] = np.copy(ainit)
for j in range(nsteps):
    amat[j+1,:] = prop @ amat[j,:]

# compute the wave function in space from each "a" vector
# do it all at once using matrix multiplication!
# psi2 = (np.abs(amat @ convmat))**2

Xind, Yind = np.meshgrid(np.arange(2*(2*nmax+1)-1),np.arange(2*(2*nmax+1)-1))
indices = np.stack([Xind.flatten(), Yind.flatten()]).T
m = 2*nmax+1
n = 2*nmax+1
a = np.array((-1)*np.arange(0,m)).reshape(m,1) 
b = np.array([np.arange(m-1,m+n-1 ),])
indx = jnp.array(a + b)

# HSB: need to keep track of which data is in JAX and which data isn't
jainit = jnp.array(ainit)
jamat = jnp.array(amat)

def adjgrad(x):
    x = lax.stop_gradient(x)
    m = 2*nmax + 1
    thetahatR = x[:m]
    thetahatI = jnp.concatenate([jnp.array([0.0]), x[m:]])
    thetahat = thetahatR + 1j*thetahatI
    
    w = jnp.concatenate([jnp.flipud(jnp.conj(thetahat)), thetahat[1:]])
    vhatmat = w[indx]
    
    # Hamiltonian matrix 
    hhatmat = kmat + vhatmat
    
    # eigendecomposition and compute propagator
    hatspec, hatstates = jnp.linalg.eigh(hhatmat)
    hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    
    #solve *forward* problem
    ahatmat = jnp.concatenate([jnp.expand_dims(jainit,0), jnp.zeros((nsteps, 2*nmax+1))])
    def forstep(j, loopamat):
        return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

    ahatmat = lax.fori_loop(0, nsteps, forstep, ahatmat)

    # compute objective function
    resid = ahatmat - jamat
    obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))

    # solve *adjoint* problem
    lambmat = jnp.concatenate([jnp.zeros((nsteps, 2*nmax+1)), jnp.expand_dims(ahatmat[nsteps,:] - jamat[nsteps,:],0)])
    def adjstep(j, looplamb):
        t = nsteps - j - 1
        return looplamb.at[t].set( ahatmat[t,:] - jamat[t,:] + hatpropH @ looplamb[t+1,:] )

    lambmat = lax.fori_loop(0, nsteps, adjstep, lambmat)

    # Compute the gradients
    # Most of this stuff is math that computes the directional derivative of the matrix exponential,
    # the part of the derivation above where we see "\partial \exp(Z) / \partial Z \cdot A"
    # for some matrix A.
    # All of this code has been checked against JAX autograd to make sure it is computing gradients correctly.
    
    offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
    expspec = jnp.exp(-1j*dt*hatspec)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(hatspec, hatspec)
    denom = offdiagmask * (-1j*dt)*(s1 - s2) + jnp.eye(m)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
    
    myeye = jnp.eye(m)
    wsR = jnp.hstack([jnp.fliplr(myeye), myeye[:,1:]]).T
    ctrmatsR = wsR[indx]
    prederivamatR = jnp.einsum('ij,jkm,kl->ilm',hatstates.conj().T,ctrmatsR,hatstates) 
    derivamatR = prederivamatR * jnp.expand_dims(mask,2)
    alldmatreal = -1j*dt*jnp.einsum('ij,jkm,kl->mil',hatstates,derivamatR,hatstates.conj().T)
    
    wsI = 1.0j*jnp.hstack([-jnp.fliplr(myeye), myeye[:,1:]])
    wsI = wsI[1:,:]
    wsI = wsI.T
    ctrmatsI = wsI[indx]
    prederivamatI = jnp.einsum('ij,jkm,kl->ilm',hatstates.conj().T,ctrmatsI,hatstates) 
    derivamatI = prederivamatI * jnp.expand_dims(mask,2)
    alldmatimag = -1j*dt*jnp.einsum('ij,jkm,kl->mil',hatstates,derivamatI,hatstates.conj().T)
    
    prederivamats = jnp.concatenate([prederivamatR, prederivamatI],axis=2)
    alldmat = jnp.vstack([alldmatreal, alldmatimag])

    # compute all entries of the gradient at once
    gradients = jnp.real(jnp.einsum('ij,ajk,ik->a', jnp.conj(lambmat[1:,:]), alldmat, ahatmat[:-1,:]))

    return obj, gradients

def adjhess(x):
    x = lax.stop_gradient(x)
    m = 2*nmax + 1
    thetahatR = x[:m]
    thetahatI = jnp.concatenate([jnp.array([0.0]), x[m:]])
    thetahat = thetahatR + 1j*thetahatI
    
    w = jnp.concatenate([jnp.flipud(jnp.conj(thetahat)), thetahat[1:]])
    vhatmat = w[indx]
    
    # Hamiltonian matrix 
    hhatmat = kmat + vhatmat
    
    # eigendecomposition and compute propagator
    hatspec, hatstates = jnp.linalg.eigh(hhatmat)
    hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    
    #solve *forward* problem
    ahatmat = jnp.concatenate([jnp.expand_dims(jainit,0), jnp.zeros((nsteps, 2*nmax+1))])
    def forstep(j, loopamat):
        return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

    ahatmat = lax.fori_loop(0, nsteps, forstep, ahatmat)

    # compute objective function
    resid = ahatmat - jamat
    obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))
    
    # solve *adjoint* problem
    lambmat = jnp.concatenate([jnp.zeros((nsteps, 2*nmax+1)), jnp.expand_dims(ahatmat[nsteps,:] - jamat[nsteps,:],0)])
    def adjstep(j, looplamb):
        t = nsteps - j - 1
        return looplamb.at[t].set( ahatmat[t,:] - jamat[t,:] + hatpropH @ looplamb[t+1] )

    lambmat = lax.fori_loop(0, nsteps, adjstep, lambmat)

    # Compute the gradients
    # Most of this stuff is math that computes the directional derivative of the matrix exponential,
    # the part of the derivation above where we see "\partial \exp(Z) / \partial Z \cdot A"
    # for some matrix A.
    # All of this code has been checked against JAX autograd to make sure it is computing gradients correctly.
    
    alldmat = []
    # alldmat = np.zeros((2*m-1, m, m), dtype=np.complex128)
    
    offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
    expspec = jnp.exp(-1j*dt*hatspec)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(hatspec, hatspec)
    denom = offdiagmask * (-1j*dt)*(s1 - s2) + jnp.eye(m)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
    
    myeye = jnp.eye(m)
    wsR = jnp.hstack([jnp.fliplr(myeye), myeye[:,1:]]).T
    ctrmatsR = wsR[indx]
    prederivamatR = jnp.einsum('ij,jkm,kl->ilm',hatstates.conj().T,ctrmatsR,hatstates) 
    derivamatR = prederivamatR * jnp.expand_dims(mask,2)
    alldmatreal = -1j*dt*jnp.einsum('ij,jkm,kl->mil',hatstates,derivamatR,hatstates.conj().T)
    
    wsI = 1.0j*jnp.hstack([-jnp.fliplr(myeye), myeye[:,1:]])
    wsI = wsI[1:,:]
    wsI = wsI.T
    ctrmatsI = wsI[indx]
    prederivamatI = jnp.einsum('ij,jkm,kl->ilm',hatstates.conj().T,ctrmatsI,hatstates) 
    derivamatI = prederivamatI * jnp.expand_dims(mask,2)
    alldmatimag = -1j*dt*jnp.einsum('ij,jkm,kl->mil',hatstates,derivamatI,hatstates.conj().T)
    
    prederivamats = jnp.concatenate([prederivamatR, prederivamatI],axis=2)
    alldmat = jnp.vstack([alldmatreal, alldmatimag])

    # compute all entries of the gradient at once
    gradients = jnp.real(jnp.einsum('ij,ajk,ik->a', jnp.conj(lambmat[1:,:]), alldmat, ahatmat[:-1,:]))
    
    # propagators
    hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    
    # propagate \nabla_{\theta} a
    gradamat = jnp.zeros((nsteps+1, 2*nmax+1, 4*nmax+1), dtype=np.complex128)
    def gradastep(j, loopgrada):
        return loopgrada.at[j+1].set( hatprop @ loopgrada[j,:,:] + (alldmat @ ahatmat[j,:]).T )

    gradamat = lax.fori_loop(0, nsteps, gradastep, gradamat)

    # propagate \nabla_{\theta} \lambda
    alldmatH = np.transpose(alldmat.conj(),axes=(2,1,0))
    gradlamb = jnp.concatenate([jnp.zeros((nsteps, 2*nmax+1, 4*nmax+1)), jnp.expand_dims(gradamat[nsteps],0)])
    def gradlambstep(j, loopgradlamb):
        t = nsteps - j - 1
        term1 = hatpropH @ loopgradlamb[t+1]
        term2 = jnp.einsum('ijk,j->ik', alldmatH, lambmat[t+1,:])
        return loopgradlamb.at[t].set( gradamat[t,:,:] + term1 + term2 )

    gradlamb = lax.fori_loop(0, nsteps, gradlambstep, gradlamb)

    hesspt1 = jnp.real(jnp.einsum('ijl,ajk,ik->al', jnp.conj(gradlamb[1:,:,:]), alldmat, ahatmat[:-1,:]))
    hesspt2 = jnp.real(jnp.einsum('ij,ajk,ikl->al', jnp.conj(lambmat[1:,:]), alldmat, gradamat[:-1,:,:]))
    res = purejaxhess(hatspec, -1j*dt*jnp.transpose(prederivamats,[2,0,1]))
    hesspt3 = jnp.real(jnp.einsum('ci,ij,abjk,lk,cl->ab',jnp.conj(lambmat[1:,:]),hatstates,res,jnp.conj(hatstates),ahatmat[:-1,:],optimize=True))
    hess = hesspt1 + hesspt2 + hesspt3
    
    return obj, gradients, hess

def purejaxhess(dvec, alldmat):
    dvec = lax.stop_gradient(dvec)
    alldmat = lax.stop_gradient(alldmat)
    jd = jnp.array(-1j*dt*dvec)
    jed = jnp.exp(jd)
    jedi = jnp.expand_dims(jed,1)
    jedj = jnp.expand_dims(jed,0)
    jdi = jnp.expand_dims(jd,1)
    jdj = jnp.expand_dims(jd,0)
    mask = jnp.expand_dims(jnp.eye(n), [0,1])
    out = jnp.expand_dims(alldmat,1)*jnp.expand_dims(alldmat,0)*mask*jedi
    jtmp = -jedi + jdi*jedi - jdj*jedi + jedj
    jtmp *= ((jdi!=jdj)*(jdi-jdj)**(-2))
    jtmp2 = jnp.expand_dims(alldmat,1).transpose([0,1,3,2]) * jnp.expand_dims(alldmat,0) 
    jtmp2 += jnp.expand_dims(alldmat,1) * jnp.expand_dims(alldmat,0).transpose([0,1,3,2])
    jtmp2 *= jtmp
    out += jnp.expand_dims(jnp.sum(jtmp2,axis=3),2) * mask
    
    unmask = 1-jnp.eye(n)
    # bmat[thread_number,i,i]
    # column vector with diagonal * amat .... bmat[thread_number,i,i]*amat[thread_number,i,k]
    # bmat * row vector with diagonal ....... bmat[thread_number,i,k]*amat[thread_number,k,k]
    bmatii = jnp.expand_dims(jnp.diagonal(alldmat,axis1=1,axis2=2),[0,3]) # * mask
    bmatkk = jnp.expand_dims(jnp.diagonal(alldmat,axis1=1,axis2=2),[0,2]) # * mask
    amatik = jnp.expand_dims(alldmat,1)
    bmatik = jnp.expand_dims(alldmat,0)
    amatkk = jnp.expand_dims(jnp.diagonal(alldmat,axis1=1,axis2=2),[1,2]) #* mask
    amatii = jnp.expand_dims(jnp.diagonal(alldmat,axis1=1,axis2=2),[1,3]) #* mask
    jdk = jnp.expand_dims(jd,0)
    jedk = jnp.expand_dims(jed,0)
    jdimk = jdi-jdk
    jdimkm1 = (jdi!=jdk)*(jdimk**(-1))
    jtmp3 = -jedi + jdimk*jedi + jedk
    jtmp3 *= (jdimkm1**2)
    jtmp4 = jtmp3 * (bmatii*amatik + amatii*bmatik)
    jtmp5 = -(-jedi + jedk + jdimk*jedk)
    jtmp5 *= (jdimkm1**2)
    jtmp6 = jtmp5 * (bmatik*amatkk + amatik*bmatkk)
    out2 = jtmp4*unmask
    out2 += jtmp6*unmask
    
    term1 = jnp.einsum('bij,ij,jk,ajk->abik',alldmat,jdimkm1,unmask,alldmat)
    term1 += jnp.einsum('aij,ij,jk,bjk->abik',alldmat,jdimkm1,unmask,alldmat)
    term1 *= (-jedi*jdimkm1)
    
    term2 = jnp.einsum('bij,j,ij,ajk,jk->abik',alldmat,jed,jdimkm1,alldmat,jdimkm1)
    term2 += jnp.einsum('aij,j,ij,bjk,jk->abik',alldmat,jed,jdimkm1,alldmat,jdimkm1)
    
    term3 = jnp.einsum('bij,ij,jk,ajk->abik',alldmat,unmask,jdimkm1,alldmat)
    term3 += jnp.einsum('aij,ij,jk,bjk->abik',alldmat,unmask,jdimkm1,alldmat)
    term3 *= (-jedk*jdimkm1)
    
    bigunmask = jnp.expand_dims(unmask,[0,1])
    out3 = -bigunmask*(term1 + term2 + term3)
    
    return out+out2+out3

def justobj(x):
    m = 2*nmax + 1
    thetahatR = x[:m]
    thetahatI = jnp.concatenate([jnp.array([0.0]), x[m:]])
    thetahat = thetahatR + 1j*thetahatI
    
    # hardcoded Toeplitz so that autodiff doesn't complain
    w = jnp.concatenate([jnp.flipud(jnp.conj(thetahat)), thetahat[1:]])
    vhatmat = w[indx]
    
    # Hamiltonian matrix 
    hhatmat = kmat + vhatmat
    
    # eigendecomposition and compute propagator
    hatspec, hatstates = jnp.linalg.eigh(hhatmat)
    hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    
    #solve *forward* problem
    ahatmat = jnp.concatenate([jnp.expand_dims(jainit,0), jnp.zeros((nsteps, 2*nmax+1))])
    def forstep(j, loopamat):
        return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

    ahatmat = lax.fori_loop(0, nsteps, forstep, ahatmat)
    
    # compute only the objective function
    resid = ahatmat - jamat
    obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))
    
    return obj


thetarand = jnp.array(np.random.normal(size=4*nmax+1))

jjustobj = jit(justobj)
jadjgrad = jit(adjgrad)
jadjhess = jit(adjhess)

# JAX gradient and Hessian
jaxgrad = grad(justobj)
jjaxgrad = jit(jaxgrad)
jaxhess = jacobian(jaxgrad)
jjaxhess = jit(jaxhess)

erro1 = 0.0
errg1 = 0.0
erro2 = 0.0
errg2 = 0.0
errh2 = 0.0
numruns = 100
for i in range(numruns):
    thetarand = jnp.array(np.random.normal(size=4*nmax+1))
    # JAX guys
    obj = jjustobj(thetarand)
    grad = jjaxgrad(thetarand)
    hess = jjaxhess(thetarand)
    # compute and check errors for outputs from jadjgrad
    obj1, grad1 = jadjgrad(thetarand)
    erro1 += jnp.abs(obj - obj1)
    errg1 += jnp.mean(jnp.abs(grad - grad1))
    # compute and check errors for outputs from jadjhess
    obj2, grad2, hess2 = jadjhess(thetarand)
    erro2 += jnp.abs(obj - obj2)
    errg2 += jnp.mean(jnp.abs(grad - grad2))
    errh2 += jnp.mean(jnp.abs(hess - hess2))

print("mean |justobj - obj from grad function| = {0:.6e}".format(erro1/numruns))
print("mean |adjgrad - jaxgrad from grad function| = {0:.6e}".format(errg1/numruns))
print("mean |justobj - obj from hess function| = {0:.6e}".format(erro2/numruns))
print("mean |adjgrad - jaxgrad from hess function| = {0:.6e}".format(errg2/numruns))
print("mean |adjhess - jaxhess| = {0:.6e}".format(errh2/numruns))

