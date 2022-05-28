import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import scipy.integrate as si
import scipy.optimize as so
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from jax.config import config
config.update("jax_enable_x64", True)

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'


###############################################################
# computational parameters
###############################################################

# size of spatial domain
L = 15.0

# number of real space grid points (for plotting)
numx = 1025

# real space grid points (for plotting)
xvec = np.linspace(-L, L, numx)

# number of Fourier basis functions
numfour = 32

# number of Toeplitz elements in the Fourier representation
numtoepelms = 2 * numfour + 1

# set time-step size
dt = 1e-2  # 1e-2

# set number of time steps
# trajectory length = numts + 1
numts = 20  # 20


###############################################################
# forward problem
###############################################################

# vector of Fourier mode indices
# fournvec = -numfour,...,0,...,numfour
fournvec = np.arange(-numfour, numfour + 1)

# matrix for converting Fourier representation to real space
fourtox = np.exp(1j * np.pi * np.outer(fournvec, xvec) / L) / np.sqrt(2 * L)

# define true potential (for generating training data)
def v(z):
    # harmonic oscillator potential (should be exact for Chebyshev)
    return 0.5 * z**2
    # symmetric double well potential
    # return 2.5e-3 * (z**2 - 25)**2
    # asymmetric double well potential
    # c0 = 4.35; c1 = 9.40e-1; c2 = -3.56e-1; c3 = -4.66e-2
    # c4 = 1.46e-2; c5 = 6.76e-4; c6 = -1.26e-4; c7 = -5.43e-6
    # c8 = 4.12e-7; c9 = 1.65e-8
    # x = z + 0.8
    # return 0.5 * (c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6 + c7*x**7 + c8*x**8 + c9*x**9)
    # non-polynomial potentials
    # return np.sin(0.4 * z - 1)
    # return np.sin((0.5 * z)**2)
    # return 15 * (-np.cos(z) + np.sin((0.5 * z)**2 - 0.2 * z))
    # soft coulomb potential
    # return -1 / np.sqrt(z**2 + 0.25)

# true potential on real space grid (for plotting)
vxvec = v(xvec)

# compute the potential operator matrix, vmat
vtoeptrue = []
for thisfourn in range(numtoepelms):
    def intgrnd(x):
        return v(x) * np.exp(-1j * np.pi * thisfourn * x / L) / (2 * L)
    def rintgrnd(x):
        return intgrnd(x).real
    def iintgrnd(x):
        return intgrnd(x).imag
    vtoeptrue.append(si.quad(rintgrnd, -L, L, limit=100)[0] + 1j * si.quad(iintgrnd, -L, L, limit=100)[0])

vtoeptrue = jnp.array(vtoeptrue)
vmattrue = sl.toeplitz(r=vtoeptrue, c=np.conj(vtoeptrue))

# define initial state functions
def psi0_0(x):
    return 10 * np.exp(-((x + 3) / 4)**2) * (2.0 / np.pi)**0.25
    # return 10 * np.exp(-((x + 3) / 2)**2) * (2.0 / np.pi)**0.25

def psi0_1(x):
    return np.exp(-((x - 3) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-((x - 3) / 2)**2) * (2.0 / np.pi)**0.25

def psi0_2(x):
    # return np.exp(-x**2) * (2.0 / np.pi)**0.25
    return np.exp(-((x - 8) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-((x - 6)/4)**2) * (2.0 / np.pi)**0.25

def psi0_3(x):
    # a weird non-symmetric wavefunction
    # return np.abs(np.sin((0.15*x - 0.5)**2))
    return np.exp(-((x + 8) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-((x + 6)/4)**2) * (2.0 / np.pi)**0.25

def psi0_4(x):
    return np.exp(-((x - 12) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-(x - 11)**2) * (2.0 / np.pi)**0.25

def psi0_5(x):
    return np.exp(-((x + 12) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-(x + 11)**2) * (2.0 / np.pi)**0.25


# function for normalizing initial wave functions
# and transforming them to the Fourier representation
def mka0(psi0fn):
    # compute psi0 normalization term
    psi0fn_prob_intgrnd = lambda x: np.abs(psi0fn(x)) ** 2
    psi0fn_norm = np.sqrt(si.quad(psi0fn_prob_intgrnd, -L, L)[0])

    # normalized psi function (for integration)
    norm_psi0fn = lambda x: psi0fn(x) / psi0fn_norm

    # compute the Fourier representation of psi0fn
    a0raw = []
    for thisfourn in range (numfour + 1):
        def intgrnd(x):
            return norm_psi0fn(x) * np.exp(-1j * np.pi * thisfourn * x / L) / np.sqrt(2 * L)
        def rintgrnd(x):
            return intgrnd(x).real
        def iintgrnd(x):
            return intgrnd(x).imag
        a0raw.append(si.quad(rintgrnd, -L, L, limit=100)[0] + 1j * si.quad(iintgrnd, -L, L, limit=100)[0])

    a0 = np.concatenate([np.conjugate(np.flipud(a0raw[1:])), a0raw])
    a0 = jnp.array(a0)
    normpsi0x = norm_psi0fn(xvec)

    return a0, normpsi0x


# generate initial state vector
# pick initial un-normalized wave functions
psi0fnvec = [psi0_0, psi0_1, psi0_2, psi0_3]  # [psi0_0, psi0_1, psi0_2, psi0_3, psi0_4, psi0_5]

# run mka0
a0vec = []
normpsi0xvec = []
normpsi0recxvec = []
for thispsi0fn in psi0fnvec:
    tempa0, tempnormpsi0x = mka0(thispsi0fn)
    a0vec.append(tempa0)
    normpsi0xvec.append(tempnormpsi0x)
    normpsi0recxvec.append(tempa0 @ fourtox)


# make kinetic operator in the Fourier representation
# (this is constant for a given system)
kmat = np.diag(np.arange(-numfour, numfour + 1) ** 2 * np.pi ** 2 / (2 * L ** 2))

# Hamiltonian operator with true potential
# in the Fourier representation
hmattrue = kmat + vmattrue

# eigen-decomposition of the Hamiltonian matrix
spctrue, stttrue = jnl.eigh(hmattrue)

# compute propagator matrix
propatrue = stttrue @ jnp.diag(jnp.exp(-1j * spctrue * dt)) @ stttrue.conj().T

# propagate system starting from initial "a" state
# using the Hamiltonian constructed from the true potential
# (used for generating training data)
amattruevec = []
for thisa0 in a0vec:
    tempamat = [thisa0.copy()]
    for i in range(numts):
        tempamat.append(propatrue @ tempamat[-1])

    amattruevec.append(tempamat)

amattruevec = jnp.array(amattruevec)

# make |\psi(t)|^2 training data from amattruevec
betamatvec = []
for thisamattrue in amattruevec:
    tempbetamat = []
    for thisavectrue in thisamattrue:
        tempbetamat.append(jnp.correlate(thisavectrue, thisavectrue, 'same'))

    betamatvec.append(jnp.array(tempbetamat))

betamatvec = jnp.array(betamatvec) / jnp.sqrt(2 * L)


###############################################################
# inverse problem
###############################################################

# Toeplitz indexing matrix, used for constructing Toeplitz matrix
# from a vector setup like:
# jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
aa = (-1) * np.arange(0, numtoepelms).reshape(numtoepelms, 1)
bb = [np.arange(numtoepelms - 1, 2 * numtoepelms - 1)]
toepindxmat = np.array(aa + bb)
print(toepindxmat.shape)

# define objective function
def ampsqobject(theta):
    # theta is a vector containing the concatenation
    # of the real and imaginary parts of vmat
    # its size should be 2 * numtoepelms - 1 = 4 * numfour + 1

    # to use theta we need to first recombine the real
    # and imaginary parts into a vector of complex values
    vtoephatR = theta[:numtoepelms]
    vtoephatI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
    vtoephat = vtoephatR + 1j * vtoephatI

    # construct vmathat from complex toeplitz vector
    vmathat = jnp.concatenate([jnp.flipud(jnp.conj(vtoephat)), vtoephat[1:]])[toepindxmat]

    # Construct Hamiltonian matrix
    hmathat = kmat + vmathat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hmathat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T

    rtnobj = 0.0
    for r in range(len(a0vec)):
        thisahat = a0vec[r].copy()
        thisbetahatmat = [jnp.correlate(thisahat, thisahat, 'same') / jnp.sqrt(2 * L)]

        # propagate system starting from initial "a" state
        for i in range(numts):
            # propagate the system one time-step
            thisahat = (propahat @ thisahat)
            # calculate the amp^2
            thisbetahatmat.append(jnp.correlate(thisahat, thisahat, 'same') / jnp.sqrt(2 * L))

        # compute objective functions
        tempresid = jnp.array(thisbetahatmat) - betamatvec[r]
        thisobj = 0.5 * jnp.sum(jnp.abs(tempresid)**2)
        rtnobj += thisobj

    return rtnobj


# true potential in the form of theta (for testing purposes)
thetatrue = jnp.concatenate((jnp.real(vtoeptrue), jnp.imag(vtoeptrue[1:])))

# jit ampsqobject
jitampsqobject = jax.jit(ampsqobject)
# complie and test jitampsqobject
print(jitampsqobject(thetatrue))

# initialize theta with random coefficients close to zero
seed = 1234  # set to None for random initialization
thetarnd = 0.001 * np.random.default_rng(seed).normal(size=thetatrue.shape)
thetarnd = jnp.array(thetarnd)

# transform init theta (i.e., initvhatmat) to real space potential
vtoepinitR = thetarnd[:numtoepelms]
vtoepinitI = jnp.concatenate((jnp.array([0.0]), thetarnd[numtoepelms:]))
vtoepinit = vtoepinitR + 1j * vtoepinitI
vinitfour = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(vtoepinit[1:(numfour + 1)])), vtoepinit[:(numfour + 1)]])
vinitrec = vinitfour @ fourtox

# function for generating M and P matrix (used in adjoint method)
def mk_M_and_P(avec):
    halflen = len(avec) // 2
    padavec = jnp.concatenate((jnp.zeros(halflen), jnp.array(avec), jnp.zeros(halflen)))

    rawmat = []
    for j in range(2 * halflen + 1):
        rawmat.append(padavec[2 * halflen - j:4 * halflen + 1 - j])

    Mmat = jnp.conjugate(jnp.array(rawmat))
    Pmat = jnp.flipud(jnp.array(rawmat))

    return Mmat, Pmat

# jit mk_M_and_P
jit_mk_M_and_P = jax.jit(mk_M_and_P)

# function for computing gradients using adjoint method
def adjgrads(theta):
    # to use theta we need to first recombine the real
    # and imaginary parts into a vector of complex values
    vtoephatR = theta[:numtoepelms]
    vtoephatI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
    vtoephat = vtoephatR + 1j * vtoephatI
    # print('Shape vtoephat:', vtoephat.shape)

    # construct vmathat from complex toeplitz vector
    vmathat = jnp.concatenate([jnp.flipud(jnp.conj(vtoephat)), vtoephat[1:]])[toepindxmat]

    # Construct Hamiltonian matrix
    hmathat = kmat + vmathat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hmathat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T
    proplam = jnp.transpose(jnp.conjugate(propahat))

    # forward propagation
    ahatmatvec = []
    lammatvec = []
    for r in range(len(a0vec)):
        # propagate system starting from initial "a" state
        thisahatmat = [a0vec[r].copy()]
        thisrhomat = [jnp.correlate(thisahatmat[0], thisahatmat[0], 'same') / jnp.sqrt(2 * L)]
        thispartlammat = [jnp.zeros(numtoepelms, dtype=complex)]

        for i in range(numts):
            # propagate the system one time-step
            thisahatmat.append(propahat @ thisahatmat[-1])

            # calculate the amp^2
            thisrhomat.append(jnp.correlate(thisahatmat[-1], thisahatmat[-1], 'same') / jnp.sqrt(2 * L))

            # compute \rho^r - \beta^r
            thiserr = thisrhomat[-1] - betamatvec[r, i+1]

            # compute M and P matrix for lambda mat
            thisMmat, thisPmat = jit_mk_M_and_P(thisahatmat[-1])

            # compute part of lambda mat
            # ( 1 / \sqrt{2 L} ) * [ ( M^r )^\dagger * ( \rho^r - \beta^r )
            # + \overline{( P^r )^\dagger * ( \rho^r - \beta^r )} ]
            thispartlammat.append((thisMmat.conj().T @ thiserr + (thisPmat.conj().T @ thiserr).conj()) / jnp.sqrt(2 * L))

        ahatmatvec.append(jnp.array(thisahatmat))

        # build lammat backwards then flip at the end
        thislammat = [thispartlammat[-1]]
        for i in range(2, numts + 2):
            thislammat.append(thispartlammat[-i] + proplam @ thislammat[-1])

        lammatvec.append(jnp.flipud(jnp.array(thislammat)))

    # make lists into JAX array object
    ahatmatvec = jnp.array(ahatmatvec)
    lammatvec = jnp.array(lammatvec)


    #######################################
    # the remainder of this function is for computing the
    # gradient of the exponential matrix
    #######################################

    offdiagmask = jnp.ones((numtoepelms, numtoepelms)) - jnp.eye(numtoepelms)
    expspec = jnp.exp(-1j * dt * spchat)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(spchat, spchat)
    denom = offdiagmask * (-1j * dt) * (s1 - s2) + jnp.eye(numtoepelms)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)

    myeye = jnp.eye(numtoepelms)
    wsR = jnp.hstack([jnp.fliplr(myeye), myeye[:,1:]]).T
    ctrmatsR = wsR[toepindxmat]
    prederivamatR = jnp.einsum('ij,jkm,kl->ilm', stthat.conj().T, ctrmatsR,stthat)
    derivamatR = prederivamatR * jnp.expand_dims(mask,2)
    alldmatreal = -1j * dt * jnp.einsum('ij,jkm,kl->mil',stthat, derivamatR, stthat.conj().T)

    wsI = 1.0j * jnp.hstack([-jnp.fliplr(myeye), myeye[:,1:]])
    wsI = wsI[1:,:]
    wsI = wsI.T
    ctrmatsI = wsI[toepindxmat]
    prederivamatI = jnp.einsum('ij,jkm,kl->ilm',stthat.conj().T, ctrmatsI, stthat)
    derivamatI = prederivamatI * jnp.expand_dims(mask, 2)
    alldmatimag = -1j * dt * jnp.einsum('ij,jkm,kl->mil',stthat, derivamatI, stthat.conj().T)

    alldmat = jnp.vstack([alldmatreal, alldmatimag])

    # compute all entries of the gradient at once
    gradients = jnp.real(jnp.einsum('bij,ajk,bik->a', jnp.conj(lammatvec[:, 1:]), alldmat, ahatmatvec[:, :-1]))

    return gradients


# jist adjgrads
jitadjgrads = jax.jit(adjgrads)
# compile and test jitadjgrads
print(nl.norm(jitadjgrads(thetatrue)))

# start optimization (i.e., learning theta)
rsltadjthetarnd = so.minimize(jitampsqobject, thetarnd, jac=jitadjgrads, tol=1e-12, options={'maxiter': 1000, 'disp': True, 'gtol': 1e-15}).x

# transform learned theta (i.e., vhatmat) to real space potential
adjvtoeplearnR = rsltadjthetarnd[:numtoepelms]
adjvtoeplearnI = jnp.concatenate((jnp.array([0.0]), rsltadjthetarnd[numtoepelms:]))
adjvtoeplearn = adjvtoeplearnR + 1j * adjvtoeplearnI
adjvlearnfour = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(adjvtoeplearn[1:(numfour + 1)])), adjvtoeplearn[:(numfour + 1)]])
adjvlearnrec = adjvlearnfour @ fourtox

# plot learned potential vs true potential
plt.plot(xvec, jnp.real(adjvlearnrec), '.-', label='adj')
plt.plot(xvec, vxvec, label='truth')
plt.plot(xvec, jnp.real(vinitrec), label='init')
plt.xlabel('x')
plt.title('True Potential vs. Learned Potential')
plt.legend()
plt.show()

# plot shifted learned potential
zeroindex = len(xvec) // 2
adjdiff = np.abs(vxvec[zeroindex] - jnp.real(adjvlearnrec)[zeroindex])
plt.plot(xvec, jnp.real(adjvlearnrec) + adjdiff, '.-', label='adj')
plt.plot(xvec, vxvec, label='truth')
plt.plot(xvec, jnp.real(vinitrec), label='init')
plt.xlabel('x')
plt.title('True Potential vs. Shifted Learned Potential')
plt.legend()
plt.show()

print('l2 error of shifted adj potential:', nl.norm(jnp.real(adjvlearnrec) + adjdiff - vxvec), sep='\n')
print('l2 error of shifted and trimmed adj potential:', nl.norm(jnp.real(adjvlearnrec)[125:-125] + adjdiff - vxvec[125:-125]), sep='\n')
print('l-inf error of shifted and trimmed adj potential:', np.mean(np.abs(jnp.real(adjvlearnrec)[125:-125] + adjdiff - vxvec[125:-125])), sep='\n')