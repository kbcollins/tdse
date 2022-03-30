from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, jit, jacobian, vmap
import jax.lax

import scipy.linalg as sl
import scipy.integrate as si
import scipy.optimize as so
import time

import numpy as np

# our Fourier representation will use basis functions from n = -nmax to n = nmax
nmax = 8

# size of spatial domain
a = 4*np.pi

def v(x):
    return np.sin(x)

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
dt = 0.01
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

def adjgrad(x):
    x = jax.lax.stop_gradient(x)
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
    
    # propagate the "a" vector of coefficients forward in time
    # in other words, solve the *forward* problem
    ahatmat = [jnp.array(ainit)]
    for j in range(nsteps):
        ahatmat.append( hatprop @ ahatmat[j] )
    
    ahatmat = jnp.stack(ahatmat)
    
    # propagate the "lambda" vector of coefficients backward in time
    # in other words, solve the *adjoint* problem
    lambmat = [ahatmat[nsteps,:] - amat[nsteps,:]]
    
    itr = 0
    for j in range(nsteps-1,-1,-1):
        lambmat.append( ahatmat[j,:] - amat[j,:] + hatpropH @ lambmat[itr] )
        itr += 1
        
    lambmat = jnp.flipud( jnp.stack(lambmat) )    
    
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

    # compute objective function
    resid = ahatmat - amat
    obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))

    return obj, gradients

def adjhess(x):
    x = jax.lax.stop_gradient(x)
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
    
    # propagate the "a" vector of coefficients forward in time
    # in other words, solve the *forward* problem
    ahatmat = [jnp.array(ainit)]
    for j in range(nsteps):
        ahatmat.append( hatprop @ ahatmat[j] )
    
    ahatmat = jnp.stack(ahatmat)
    
    # propagate the "lambda" vector of coefficients backward in time
    # in other words, solve the *adjoint* problem
    lambmat = [ahatmat[nsteps,:] - amat[nsteps,:]]
    
    itr = 0
    for j in range(nsteps-1,-1,-1):
        lambmat.append( ahatmat[j,:] - amat[j,:] + hatpropH @ lambmat[itr] )
        itr += 1
        
    lambmat = jnp.flipud( jnp.stack(lambmat) )    
    
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

    # compute objective function
    resid = ahatmat - amat
    obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))
    
    # compute all entries of the gradient at once
    gradients = jnp.real(jnp.einsum('ij,ajk,ik->a', jnp.conj(lambmat[1:,:]), alldmat, ahatmat[:-1,:]))
    
    # propagators
    hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    
    # propagate \nabla_{\theta} a
    gradamat = [jnp.zeros((2*nmax+1, 4*nmax+1))]
    for j in range(nsteps):
        gradamat.append( hatprop @ gradamat[j] + (alldmat @ ahatmat[j,:]).T )
        
    gradamat = jnp.stack(gradamat)
    
    # propagate \nabla_{\theta} \lambda
    alldmatH = np.transpose(alldmat.conj(),axes=(2,1,0))
    gradlamb = [gradamat[nsteps]]
    itr = 0
    for j in range(nsteps-1,-1,-1):
        term1 = hatpropH @ gradlamb[itr]
        term2 = jnp.einsum('ijk,j->ik', alldmatH, lambmat[j+1,:])
        gradlamb.append( gradamat[j,:,:] + term1 + term2 )
        itr += 1
    
    gradlamb = jnp.flipud( jnp.stack(gradlamb) )
    
    hesspt1 = jnp.real(jnp.einsum('ijl,ajk,ik->al', jnp.conj(gradlamb[1:,:,:]), alldmat, ahatmat[:-1,:]))
    hesspt2 = jnp.real(jnp.einsum('ij,ajk,ikl->al', jnp.conj(lambmat[1:,:]), alldmat, gradamat[:-1,:,:]))
    res = purejaxhess(hatspec, -1j*dt*jnp.transpose(prederivamats,[2,0,1]))
    hesspt3 = jnp.real(jnp.einsum('ci,ij,abjk,lk,cl->ab',jnp.conj(lambmat[1:,:]),hatstates,res,jnp.conj(hatstates),ahatmat[:-1,:],optimize=True))
    hess = hesspt1 + hesspt2 + hesspt3
    
    return obj, gradients, hess

def purejaxhess(dvec, alldmat):
    dvec = jax.lax.stop_gradient(dvec)
    alldmat = jax.lax.stop_gradient(alldmat)
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
    
    # propagate the "a" vector of coefficients as defined above
    ahatmat = [jnp.array(ainit)]
    for j in range(nsteps):
        ahatmat.append( hatprop @ ahatmat[j] )
    
    ahatmat = jnp.stack(ahatmat)
    
    # compute only the objective function
    resid = ahatmat - amat
    obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))
    
    return obj

# jit compile
jadjgrad = jit(adjgrad)
jadjhess = jit(adjhess)
jobj = jit(justobj)

# numpy wrappers
def npadjgrad(theta):
    x = jnp.array(theta)
    obj, grad = jadjgrad(x)
    return np.squeeze(np.array(obj)), np.array(grad)

def npadjhess(theta):
    x = jnp.array(theta)
    _, _, hess = jadjhess(x)
    return np.array(hess)    

# random initialization
theta0 = 0.01*jnp.array( np.random.normal(size=4*nmax+1) )

# optimize
res = so.minimize(fun=npadjgrad, x0=theta0, method='trust-krylov', jac=True, hess=npadjhess, tol=1e-8, options={'disp': True})

# true theta
truethetaR = jnp.real(vrow)
truethetaI = jnp.imag(vrow[1:])
xtrue = jnp.concatenate([truethetaR, truethetaI])

# print final objective function value
# and norm of gradient
finalobj, finalgrad = npadjgrad(res.x)
print("")
print("Final value of objective:")
print(finalobj)
print("")
print("Final || gradient ||:")
print(np.linalg.norm(finalgrad))

# error of learned theta
print("")
print("|| thetaTRUE - thetaLEARNED ||")
print(np.linalg.norm(xtrue - res.x))


