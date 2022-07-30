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

def v(x):
    return 0.5*x**2

# let's take our new class for a spin
fourmodel = linearbases.fourier(biga, 1025, nmax)
# truemodel = linearbases.chebyshev(biga, 1025, nmax, 5)
truemodel = fourmodel
truemodel.represent(v)
jrepmat = jnp.array(truemodel.grad())
print(jrepmat.shape)
repsize = jrepmat.shape[2] - 1
xvec, vvec = truemodel.vx()


# Hamiltonian matrix 
hmat = fourmodel.kmat() + truemodel.vmat()

# check whether Hamiltonian is Hermitian
print(np.linalg.norm( hmat - np.conj(hmat.T) ))

# eigendecomposition
spec, states = np.linalg.eigh(hmat)

# check whether this diagonalizes hmat
# note that if this is close to zero it means that
# hmat = states @ np.diag(spec) @ np.conj(states).T
print(np.linalg.norm( hmat @ states - states @ np.diag(spec) ))

# check whether we have a unitary matrix
print(np.linalg.norm( states @ np.conj(states).T - np.eye(2*nmax+1) ))

# pick out ground state
ordering = np.argsort(spec)
groundstate = ordering[1]
wavefn = states[:,groundstate] @ fourmodel.fourtoxmat 

# check normalization
print(np.sum(np.abs(wavefn)**2 * fourmodel.dx))

# rounded box function
def psi0(x):
    return (1.0 + np.tanh((1 - x**2)/0.5))/2.58046

# use the class again to represent initial condition!
initcond = linearbases.fourier(biga, 1025, nmax)
ainit = initcond.represent(psi0)

####################
print('-->ainit:', ainit)
####################

# set the time step and compute the propagator matrix
# note that we are reusing the spec, states eigendecomposition of hmat computed above
dt = 0.001
prop = states @ np.diag(np.exp(-1j*spec*dt)) @ np.conj(states.T)

# propagate the "a" vector of coefficients as defined above
nsteps = 200
amat = np.zeros((nsteps+1, 2*nmax+1), dtype=np.complex128)
amat[0,:] = np.copy(ainit)
for j in range(nsteps):
    amat[j+1,:] = prop @ amat[j,:]

# HSB: need to keep track of which data is in JAX and which data isn't
jainit = jnp.array(ainit)
jamat = jnp.array(amat)
jkmat = jnp.array(fourmodel.kmat())

# parallelize the built-in correlate function over first axes
vcorr = vmap(jnp.correlate, in_axes=(0,0,None))

# compute Fourier coefficients of |amat|^2
jbetamat = vcorr(amat, amat, 'same') / jnp.sqrt(2 * biga)

def justobj(x, ic):
    # potential matrix
    vhatmat = jrepmat @ x
    
    # Hamiltonian matrix 
    hhatmat = jkmat + vhatmat
    
    # eigendecomposition and compute propagator
    hatspec, hatstates = jnp.linalg.eigh(hhatmat)
    hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*dt)) @ jnp.conj(hatstates.T)
    
    #solve *forward* problem
    ahatmat = jnp.concatenate([jnp.expand_dims(ic,0), jnp.zeros((nsteps, 2*nmax+1))])
    def forstep(j, loopamat):
        return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

    ahatmat = lax.fori_loop(0, nsteps, forstep, ahatmat)
    rhomat = vcorr(ahatmat, ahatmat, 'same') / jnp.sqrt(2 * biga)
    
    # compute only the objective function
    resid = rhomat - jbetamat
    obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))
    
    return obj

########################################
# kbc

realjainit = jnp.concatenate([ainit.real, ainit.imag])
print('-->Shape realjainit:', realjainit.shape)


def objrealic(x, realic):
    # recombine real and imaginary parts of ic
    print('-->realic:', realic)
    print('-->realic[:2*nmax + 1]:', realic[:2*nmax + 1])
    print('-->realic[2*nmax + 1:]:', realic[2*nmax + 1:])
    ic = realic[:2*nmax + 1] + 1j*realic[2*nmax + 1:]

    # potential matrix
    vhatmat = jrepmat @ x

    # Hamiltonian matrix
    hhatmat = jkmat + vhatmat

    # eigendecomposition and compute propagator
    hatspec, hatstates = jnp.linalg.eigh(hhatmat)
    hatprop = hatstates @ jnp.diag(jnp.exp(-1j * hatspec * dt)) @ jnp.conj(hatstates.T)

    # solve *forward* problem
    ahatmat = jnp.concatenate([jnp.expand_dims(ic, 0), jnp.zeros((nsteps, 2 * nmax + 1))])

    def forstep(j, loopamat):
        return loopamat.at[j + 1].set(hatprop @ loopamat[j, :])

    ahatmat = lax.fori_loop(0, nsteps, forstep, ahatmat)
    rhomat = vcorr(ahatmat, ahatmat, 'same') / jnp.sqrt(2 * biga)

    # compute only the objective function
    resid = rhomat - jbetamat
    obj = 0.5 * jnp.real(jnp.sum(jnp.conj(resid) * resid))

    return obj

jitobjrealic = jit(objrealic)
jgradobjrealic = jit(grad(objrealic))

def compgradhess(x, ic):
    ########################################
    # I'm using this code as a quick and dirty way to get
    # the propagator matrix and resid. If my Hessian works
    # there are better ways to do this.
    # potential matrix
    vhatmat = jrepmat @ x
    # Hamiltonian matrix
    hhatmat = jkmat + vhatmat

    # eigendecomposition and compute propagator
    hatspec, hatstates = jnp.linalg.eigh(hhatmat)
    hatprop = hatstates @ jnp.diag(jnp.exp(-1j * hatspec * dt)) @ jnp.conj(hatstates.T)

    # solve *forward* problem
    ahatmat = jnp.concatenate([jnp.expand_dims(ic, 0), jnp.zeros((nsteps, 2 * nmax + 1))])

    def forstep(j, loopamat):
        return loopamat.at[j + 1].set(hatprop @ loopamat[j, :])

    ahatmat = lax.fori_loop(0, nsteps, forstep, ahatmat)
    rhomat = vcorr(ahatmat, ahatmat, 'same') / jnp.sqrt(2 * biga)

    # compute only the objective function
    resid = rhomat - jbetamat
    # print('-->Shape resid:', resid.shape)
    print('-->resid[50]:', resid[50])
    ########################################

    p1mat = jnp.exp(-1j * hhatmat * dt)
    # print('-->Shape pmat:', pmat.shape)

    alpha = 1 / np.sqrt(2 * biga)
    # print(alpha)

    # build vecs
    for j in range(nsteps + 1):
        pjmat = p1mat ** j

        # this essentially is the propagation of a0
        # again this is just a dirty way to get this term
        # for now
        ajvec = pjmat @ ainit

        dJ1 = np.zeros(2*nmax + 1)
        dJ2 = np.zeros(2 * nmax + 1)
        A = np.zeros((2*nmax + 1, 2*nmax + 1))
        B = np.zeros((2 * nmax + 1, 2 * nmax + 1))
        C = np.zeros((2 * nmax + 1, 2 * nmax + 1), dtype=complex)
        D = np.zeros((2 * nmax + 1, 2 * nmax + 1), dtype=complex)
        G = np.zeros((2 * nmax + 1, 2 * nmax + 1), dtype=complex)
        for s in range(2*nmax + 1):
            psvec = pjmat.T[s]
            # in notes correlation writen like (v \star a)
            # numpy.correlate(a, v, mode=)
            corrpsaj = jnp.correlate(ajvec, psvec, mode='same')
            # print('-->corrpsaj:', corrpsaj)
            corrajps = jnp.correlate(psvec, ajvec, mode='same')
            # print('-->corrajps:', corrajps)
            dJ1 += alpha * np.real(np.transpose(np.conj(resid[j])) @ (corrpsaj + corrajps))
            dJ2 += -alpha * np.imag(np.transpose(np.conj(resid[j])) @ (-corrpsaj + corrajps))
            # for r in range(2*nmax + 1):
            #     prvec = pjmat.T[r]
            #     # in notes correlation writen like (v \star a)
            #     # numpy.correlate(a, v, mode=)
            #     corrpraj = np.correlate(ajvec, prvec, mode='same')
            #     corrajpr = np.correlate(prvec, ajvec, mode='same')
            #     corrpspr = np.correlate(prvec, psvec, mode='same')
            #     corrprps = np.correlate(psvec, prvec, mode='same')
            #     A[r, s] += alpha * np.real(np.transpose(np.conj(resid[j])) @ (corrpspr + corrprps))
            #     B[r, s] += alpha * np.real(1j * np.transpose(np.conj(resid[j])) @ (corrpspr - corrprps))
            #     termc1 = np.transpose(np.conj(corrpraj + corrajpr)) @ (-corrpsaj + corrajps)
            #     termc2 = np.transpose(np.conj(-corrpraj + corrajpr)) @ (corrpsaj + corrajps)
            #     C[r, s] += 1j * alpha**2 * (termc1 - termc2) / 2
            #     D[r, s] += alpha**2 * np.transpose(np.conj(corrpraj + corrajpr)) @ (corrpsaj + corrajps)
            #     G[r, s] += alpha**2 * np.transpose(np.conj(-corrpraj + corrajpr)) @ (-corrpsaj + corrajps)

    gradJ = np.concatenate([dJ1, dJ2])
    hessJ = np.block([[A + D, -B + C], [B + C, A + G]])
    # print('-->Shape blockmat:', blockmat.shape)

    return gradJ, hessJ
########################################

print("justobj at true theta: ")
print(justobj(truemodel.gettheta(), jainit))

jjustobj = jit(justobj)

# JAX gradient and Hessian w.r.t. theta
jaxgrad = grad(justobj, 0)
jjaxgrad = jit(jaxgrad)
jaxhess = jacobian(jaxgrad, 0)
jjaxhess = jit(jaxhess)

# JAX gradient w.r.t. initial condition
jaxdinit = grad(justobj, 1)
jjaxdinit = jit(jaxdinit)

# JAX mixed hessian w.r.t. initial conditions and theta
def ridinit(theta, ic):
    tmp = jaxdinit(theta, ic)
    return jnp.stack([jnp.real(tmp), jnp.imag(tmp)])

jaxmixed = jacobian(ridinit, 0)
jjaxmixed = jit(jaxmixed)

# finite-difference Hessian for fun and profit
def jaxhinit(theta, ic):
    eps = 1e-10
    n = ic.shape[0]
    myhess = []
    myeye = jnp.eye(n)
    for j in range(n):
        icp = ic + eps*myeye[j]
        icm = ic - eps*myeye[j]
        tmp = (jaxdinit(theta, icp) - jaxdinit(theta, icm))/(2*eps)
        myhess.append( tmp )

    return jnp.array(myhess)

jjaxhinit = jit(jaxhinit)

testadj = ls.learnschro(biga, dt, nmax, obj='A2', ic=jainit, wfdata=jamat, kinmat=jkmat)

erro1 = 0.0
errg1 = 0.0
errd1 = 0.0
errd2 = 0.0
erro2 = 0.0
errg2 = 0.0
errh2 = 0.0
errm2 = 0.0
fderr = 0.0
numruns = 1  # 10
for i in range(numruns):
    thetarand = jnp.array(np.random.normal(size=repsize+1))
    # adjmodel = linearbases.chebyshev(biga, 1025, nmax, repsize, theta=thetarand)
    adjmodel = linearbases.fourier(biga, 1025, nmax, theta=thetarand)
    jvhatmat = jnp.array(adjmodel.vmat())
    jctrmats = jnp.array(adjmodel.grad())
    # JAX guys

    #####################################
    obj = jjustobj(thetarand, jainit)
    objRic = jitobjrealic(thetarand, realjainit)
    dobjrealic = jgradobjrealic(thetarand, realjainit)
    #####################################

    grad = jjaxgrad(thetarand, jainit)
    hess = jjaxhess(thetarand, jainit)
    dinit = jjaxdinit(thetarand, jainit)
    rimixed = jjaxmixed(thetarand, jainit)
    mixed = rimixed[0] + 1j*rimixed[1]
    hinit = jjaxhinit(thetarand, jainit)

    ########################################
    # kbc
    print('-->objRic:', objrealic(thetarand, realjainit))
    print('-->Error objRic:', np.linalg.norm(obj - objRic))
    print('-->Shape dobjrealic:', dobjrealic.shape)
    print('-->dobjrealic:', dobjrealic)

    dJ, hJ = compgradhess(truemodel.gettheta(), jainit)
    print('-->Shape dJ:', dJ.shape)
    print('-->dJ:', dJ)

    # print('-->Shape hJ:', hJ.shape)
    # print('-->hinit:', hinit)
    # print('-->hJ:', hJ)

    # replace (nsteps+1)*jnp.eye(jainit.shape[0]) with Hessian
    # print(hinit)
    fderr += jnp.mean(jnp.abs(hinit - hJ))
    # fderr += jnp.mean(jnp.abs(hinit - (nsteps+1)*jnp.eye(jainit.shape[0])))

    print(jnp.abs(hinit - hJ))
    ########################################

    # compute and check errors for outputs from jadjgrad
    obj1, grad1, dinit1 = testadj.jadjgrad(jvhatmat, jctrmats)
    erro1 += jnp.abs(obj - obj1)
    errg1 += jnp.mean(jnp.abs(grad - grad1))
    errd1 += jnp.mean(jnp.abs(dinit - dinit1))
    # compute and check errors for outputs from jadjhess
    obj2, grad2, dinit2, hess2, mixed2 = testadj.jadjhess(jvhatmat, jctrmats)
    erro2 += jnp.abs(obj - obj2)
    errg2 += jnp.mean(jnp.abs(grad - grad2))
    errh2 += jnp.mean(jnp.abs(hess - hess2))
    errd2 += jnp.mean(jnp.abs(dinit - dinit2))
    errm2 += jnp.mean(jnp.abs(mixed - mixed2))

print("mean |justobj - obj from grad function| = {0:.6e}".format(erro1/numruns))
print("mean |adjgrad - jaxgrad from grad function| = {0:.6e}".format(errg1/numruns))
print("mean |justobj - obj from hess function| = {0:.6e}".format(erro2/numruns))
print("mean |adjgrad - jaxgrad from hess function| = {0:.6e}".format(errg2/numruns))
print("mean |adjhess - jaxhess| = {0:.6e}".format(errh2/numruns))

print("")
print("mean |adjdinit - jaxdinit from grad function| = {0:.6e}".format(errd1/numruns))
print("mean |adjdinit - jaxdinit from hess function| = {0:.6e}".format(errd2/numruns))
print("")
print("mean |adjmixed - jaxmixed from hess function| = {0:.6e}".format(errm2/numruns))
print("")
print("mean |fdIChess - trueIChess| = {0:.6e}".format(fderr/numruns))
print("")

