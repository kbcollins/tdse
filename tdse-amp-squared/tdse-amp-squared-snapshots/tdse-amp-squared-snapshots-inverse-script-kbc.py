import sys
import pathlib
import numpy as np
import numpy.linalg as nl
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
# set directory to load data from
###############################################################

# get path to directory containing amat from command line
cmdlinearg = sys.argv[1]
print('Command line argument:', cmdlinearg)

# transform commandline argument to path object
cwddir = pathlib.Path(cmdlinearg)
print('Current working directory:', cwddir)


###############################################################
# load computational environment
###############################################################

L, numx, numfour, dt, numts = np.load(cwddir / 'cmpprm.npy')
numx = int(numx)
numfour = int(numfour)
numts = int(numts)

# load state variables
a0vec = np.load(cwddir / 'a0vec.npy')
amattruevec = np.load(cwddir / 'amattruevec.npy')

print('Computational environment loaded.')


###############################################################
# recreate variables from loaded data
###############################################################

# real space grid points (for plotting)
xvec = np.linspace(-L, L, numx)

# vector of Fourier mode indices
# fournvec = -numfour,...,0,...,numfour
fournvec = np.arange(-numfour, numfour + 1)

# matrix for converting Fourier representation to real space
# use like realspacevec = fourspacevec @ fourtox
fourtox = np.exp(1j * np.pi * np.outer(fournvec, xvec) / L) / np.sqrt(2 * L)

# number of Toeplitz elements in the Fourier representation
numtoepelms = 2 * numfour + 1

# make kinetic operator in the Fourier representation
# (this is constant for a given system)
kmat = np.diag(np.arange(-numfour, numfour + 1) ** 2 * np.pi ** 2 / (2 * L ** 2))


###############################################################
# make |\psi(t)|^2 training data from amattruevec
###############################################################

print('Starting inverse problem.')

betamatvec = []
for thisamattrue in amattruevec:
    tempbetamat = []
    for thisavectrue in thisamattrue:
        tempbetamat.append(jnp.correlate(thisavectrue, thisavectrue, 'same'))

    betamatvec.append(jnp.array(tempbetamat))

betamatvec = jnp.array(betamatvec) / jnp.sqrt(2 * L)

print('Training data generated.')


###############################################################
# make Toeplitz indexing matrix
###############################################################

# use toepindexmat to construct Toeplitz matrix from a vector
# of the form jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
aa = (-1) * np.arange(0, numtoepelms).reshape(numtoepelms, 1)
bb = [np.arange(numtoepelms - 1, 2 * numtoepelms - 1)]
toepindxmat = np.array(aa + bb)
# print(toepindxmat.shape)


###############################################################
# define objective function
###############################################################

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


###############################################################
# theta
###############################################################

# true potential in the form of theta (for testing purposes)
# thetatrue = jnp.concatenate((jnp.real(vtruetoep), jnp.imag(vtruetoep[1:])))

# initialize theta with random coefficients close to zero
seed = 1234  # set to None for random initialization
thetarnd = 0.001 * np.random.default_rng(seed).normal(size=numtoepelms * 2 - 1)
thetarnd = jnp.array(thetarnd)

# transform randtheta theta (i.e., initvhatmat) to real space potential
vtoepinitR = thetarnd[:numtoepelms]
vtoepinitI = jnp.concatenate((jnp.array([0.0]), thetarnd[numtoepelms:]))
vtoepinit = vtoepinitR + 1j * vtoepinitI
vinitfour = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(vtoepinit[1:(numfour + 1)])), vtoepinit[:(numfour + 1)]])
# print('Shape vinitfour:', vinitfour.shape)
# print('Shape fourtox:', fourtox.shape)
vinitrec = vinitfour @ fourtox


###############################################################
# adjoint method for computing gradient
###############################################################

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


###############################################################
# jit ampsquaredobjective and adjgrads
###############################################################

# jit ampsquaredobjective
jitampsqobject = jax.jit(ampsqobject)
# complie jitampsquaredobjective
print('jitampsquaredobjective(thetarnd) =', jitampsqobject(thetarnd))

# jit adjgrads
jitadjgrads = jax.jit(adjgrads)
# compile jitadjgrads
print('nl.norm(jitadjgrads(thetarnd)) =', nl.norm(jitadjgrads(thetarnd)))


###############################################################
# start learning
###############################################################

# start optimization (i.e., learning theta)
rsltadjthetarnd = so.minimize(jitampsqobject, thetarnd, jac=jitadjgrads, tol=1e-12, options={'maxiter': 4000, 'disp': True, 'gtol': 1e-15}).x


###############################################################
# make plot of learned potential
###############################################################

# transform learned theta (i.e., vhatmat) to real space potential
adjvtoeplearnR = rsltadjthetarnd[:numtoepelms]
adjvtoeplearnI = jnp.concatenate((jnp.array([0.0]), rsltadjthetarnd[numtoepelms:]))
adjvtoeplearn = adjvtoeplearnR + 1j * adjvtoeplearnI
adjvlearnfour = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(adjvtoeplearn[1:(numfour + 1)])), adjvtoeplearn[:(numfour + 1)]])
adjvlearnrec = adjvlearnfour @ fourtox

# plot learned potential
plt.plot(xvec, jnp.real(adjvlearnrec), '.-', label='adj')
plt.plot(xvec, jnp.real(vinitrec), label='randtheta')
plt.xlabel('x')
plt.title('Learned Potential')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_learned_potential.pdf', format='pdf')
plt.close()

# eventually want to compare snapshot of evolution against evolution generated
# from learned potential


###############################################################
# propagate a0 with learned potential
###############################################################

# first recombine learned theta into vector of complex values
vtoeplearnedR = rsltadjthetarnd[:numtoepelms]
vtoeplearnedI = jnp.concatenate((jnp.array([0.0]), rsltadjthetarnd[numtoepelms:]))
vtoeplearned = vtoeplearnedR + 1j * vtoeplearnedI

# construct vmatlearned from complex toeplitz vector
vmatlearned = jnp.concatenate([jnp.flipud(jnp.conj(vtoeplearned)), vtoeplearned[1:]])[toepindxmat]

# Hamiltonian operator with true potential
# in the Fourier representation
hmatlearned = kmat + vmatlearned

# eigen-decomposition of the Hamiltonian matrix
spclearned, sttlearned = jnl.eigh(hmatlearned)

# compute propagator matrix
propalearned = sttlearned @ jnp.diag(jnp.exp(-1j * spclearned * dt)) @ sttlearned.conj().T

# propagate system starting from initial "a" state
# using the Hamiltonian constructed from the true potential
# (used for generating training data)
amatlearnedvec = []
for thisa0 in a0vec:
    tempamat = [thisa0.copy()]
    for i in range(numts):
        tempamat.append(propalearned @ tempamat[-1])

    amatlearnedvec.append(tempamat)

amatlearnedvec = jnp.array(amatlearnedvec)

print('L2 error of amat:', nl.norm(amattruevec - amatlearnedvec, axis=0))

# plot of real part of last state of system propagated with learned potential vs.
# last state of amat
for i in range(len(amattruevec)):
    psiTlearned = amatlearnedvec[i, -1] @ fourtox
    psiTtrue = amattruevec[i, -1] @ fourtox
    plt.plot(xvec, jnp.real(psiTlearned), '.-', label='learned')
    plt.plot(xvec, jnp.real(psiTtrue), label='truth')
    plt.xlabel('x')
    plt.title('Real Part of Final State - Learned vs. Truth')
    plt.legend()
    # plt.show()
    plt.savefig(cwddir / f'graph_real_part_last_state_learned_vs_truth_{i}.pdf', format='pdf')
    plt.close()