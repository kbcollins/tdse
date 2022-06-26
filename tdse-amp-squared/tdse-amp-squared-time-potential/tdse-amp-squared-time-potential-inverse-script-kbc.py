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
# Given a continue time-dependent potential, we assume that we
# can reasonably approximate the potential as a finite set of
# static potentials.
#
# We will learn the potentials one at a time, in chronological
# order. The training data is given as one complete trajectory,
# so we will need to first take a chunk of the data, feed the
# into our learning method, and store the learned potential.
# This way most of the code remains the same and all of the
# infrastructure for handling time-dependent potentials is like
# a wrapper around the static potential code.
#
# Internally, the system's wave function in the Fourier
# representation is propagated, and from the wave function the
# amplitude squared is computed, which is then compared against
# the training data. The initial wave function, psi0, is given
# by the problem and none of the intermediate wave functions
# are stored. For the time-dependent code to work, we will need
# to store the wave function from the previous time step.
###############################################################


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

L, numx, numfour, dt, numts = np.load(cwddir / 'cmpenv.npy')
numx = int(numx)
numfour = int(numfour)
numts = int(numts)

# load state variables
# a0vec = np.load(cwddir / 'a0vec.npy')
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

# construct initial state vector
a0vec = amattruevec[:, 0]
print('Shape a0vec:', a0vec.shape)

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
# Toeplitz indexing matrix
###############################################################

# toepindexmat is used to construct a Toeplitz matrix
# from a vector of the form
# jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
aa = (-1) * np.arange(0, numtoepelms).reshape(numtoepelms, 1)
bb = [np.arange(numtoepelms - 1, 2 * numtoepelms - 1)]
toepindxmat = np.array(aa + bb)
# print(toepindxmat.shape)


###############################################################
# theta
###############################################################

# true potential in the form of theta (for testing purposes)
# thetatrue = jnp.concatenate((jnp.real(vtoeptrue), jnp.imag(vtoeptrue[1:])))

# initialize theta with random coefficients close to zero
seed = 1234  # set to None for random initialization
thetarnd = 0.001 * np.random.default_rng(seed).normal(size=numtoepelms * 2 - 1)
thetarnd = jnp.array(thetarnd)

# transform init theta (i.e., initvhatmat) to real space potential
vtoepinitR = thetarnd[:numtoepelms]
vtoepinitI = jnp.concatenate((jnp.array([0.0]), thetarnd[numtoepelms:]))
vtoepinit = vtoepinitR + 1j * vtoepinitI
vinitfour = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(vtoepinit[1:(numfour + 1)])), vtoepinit[:(numfour + 1)]])
# print('Shape vinitfour:', vinitfour.shape)
# print('Shape fourtox:', fourtox.shape)
vinitrec = vinitfour @ fourtox


###############################################################
# define objective function
###############################################################

def ampsqobject(theta, thisbetamatvec):
    ###############################################################
    # this function assumes the potential, represented by theta,
    # is static for the entirety of the propagation.
    #
    # theta is a vector containing the concatenation
    # of the real and imaginary parts of vmat
    # its size should be 2 * numtoepelms - 1 = 4 * numfour + 1
    #
    # thisbetamatvec is the training data
    ###############################################################

    # first recombine the real and imaginary parts of the potential
    # into a vector of complex values
    vtoephatR = theta[:numtoepelms]
    vtoephatI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
    vtoephat = vtoephatR + 1j * vtoephatI

    # construct vmathat from complex vector vtoephat
    vmathat = jnp.concatenate([jnp.flipud(jnp.conj(vtoephat)), vtoephat[1:]])[toepindxmat]

    # Construct the Hamiltonian matrix
    hmathat = kmat + vmathat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hmathat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T

    # forward propagation loop
    rtnobj = 0.0
    for r in range(thisbetamatvec.shape[0]):
        thisahat = thisbetamatvec[r, 0].copy()
        thisbetahatmat = [jnp.correlate(thisahat, thisahat, 'same') / jnp.sqrt(2 * L)]

        # print('len(thisbetamatvec[r] =', len(thisbetamatvec[r]))
        # propagate system starting from initial "a" state
        for _ in range(len(thisbetamatvec[r]) - 1):
            # propagate the system one time-step
            thisahat = (propahat @ thisahat)
            # calculate the amp^2
            thisbetahatmat.append(jnp.correlate(thisahat, thisahat, 'same') / jnp.sqrt(2 * L))

        # compute objective functions
        tempresid = jnp.array(thisbetahatmat) - thisbetamatvec[r]
        thisobj = 0.5 * jnp.sum(jnp.abs(tempresid)**2)
        rtnobj += thisobj

    return rtnobj

# jit ampsqobject
jitampsqobject = jax.jit(ampsqobject)
# compile jitampsqobject
# print('jitampsqobject(thetarnd) =', jitampsqobject(thetarnd, a0vec, betamatvec[:, :1]))


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

# function for computing gradients using the adjoint method
def adjgrads(theta, thisbetamatvec):
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

    # forward propagation loop
    ahatmatvec = []
    lammatvec = []
    for r in range(thisbetamatvec.shape[0]):
        # propagate system starting from initial "a" state
        thisahatmat = [thisbetamatvec[r, 0].copy()]
        thisbetahatmat = [jnp.correlate(thisahatmat[0], thisahatmat[0], 'same') / jnp.sqrt(2 * L)]
        thispartlammat = [jnp.zeros(numtoepelms, dtype=complex)]

        # propagate system starting from thisa0vec state
        for _ in range(len(thisbetamatvec[r]) - 1):
            # propagate the system one time-step
            thisahat = (propahat @ thisahat)

            # calculate the amp^2
            thisbetahatmat.append(jnp.correlate(thisahat, thisahat, 'same') / jnp.sqrt(2 * L))

            # compute \betahat^r - \beta^r
            thiserr = thisbetahatmat[-1] - thisbetamatvec[r, i+1]

            # compute M and P matrix for lambda mat
            thisMmat, thisPmat = jit_mk_M_and_P(thisahatmat[-1])

            # compute part of lambda mat
            # ( 1 / \sqrt{2 L} ) * [ ( M^r )^\dagger * ( \rho^r - \beta^r )
            # + \overline{( P^r )^\dagger * ( \rho^r - \beta^r )} ]
            thispartlammat.append((thisMmat.conj().T @ thiserr + (thisPmat.conj().T @ thiserr).conj()) / jnp.sqrt(2 * L))

        # store compute ahatmat
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

# jit adjgrads
jitadjgrads = jax.jit(adjgrads)
# compile jitadjgrads
# print('nl.norm(jitadjgrads(thetarnd)) =', nl.norm(jitadjgrads(thetarnd)))


###############################################################
# learning loop
#   - take the training data (betamatvec) and split
#    it into sections
#   - each section is assumed to be influenced by a
#    static potential
###############################################################

numsec = 1
print('numsec =', numsec)

seclen = (betamatvec.shape[1]) // numsec
print('Shape betamatvec:', betamatvec.shape)
print('seclen =', seclen)

thisavec = a0vec.copy()
thetavec = []

for i in range(numsec):
    print(f'Starting section {i * seclen}:{(i + 1) * seclen}.')
    thisbetamatvec = betamatvec[:, i*seclen:(i + 1)*seclen]

    # optimize (i.e., learning theta)
    #
    # fun: callable - the objective function to be minimized.
    # fun(x, *args) -> float
    # where x is a 1-D array with shape (n,) and args in a tuple is a
    # tuple of the fixed parameters needed to completely specify the
    # function.
    #
    # x0: ndarray, shape (n,) - Initial guess. Array of real elements of size (n,),
    # where n is the number of independent variables
    #
    # args: tuple, optional - extra agruments passed to the objective
    # function and its derivatives (fun, jac and hess functions)
    #
    # jac: callable, optional - Method for computing the gradient vector.
    # If it is a callable, it should be a function that returns the gradient
    # vector
    # jac(x, *args) -> array_like, shape (n,)
    # where x is an array with shape (n,) and args is a tuple with the fixed
    # parameters.
    #
    thisresult = so.minimize(fun=jitampsqobject, x0=thetarnd, args=(thisbetamatvec), jac=jitadjgrads, tol=1e-12, options={'maxiter': 4000, 'disp': True, 'gtol': 1e-15}).x
    thetavec.append(thisresult)


    ###############################################################
    # make plot of the learned potential
    ###############################################################

    # transform learned theta (i.e., vhatmat) to real space potential
    adjvtoeplearnR = thetavec[-1][:numtoepelms]
    adjvtoeplearnI = jnp.concatenate((jnp.array([0.0]), thetavec[-1][numtoepelms:]))
    adjvtoeplearn = adjvtoeplearnR + 1j * adjvtoeplearnI
    adjvlearnfour = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(adjvtoeplearn[1:(numfour + 1)])), adjvtoeplearn[:(numfour + 1)]])
    adjvlearnrec = adjvlearnfour @ fourtox

    # plot learned potential
    plt.plot(xvec, jnp.real(adjvlearnrec), '.-', label=f'v{i}')
    # plt.plot(xvec, jnp.real(vinitrec), label='init')
    plt.xlabel('x')
    plt.title('Learned Potential')
    plt.legend()
    # plt.show()
    plt.savefig(cwddir / f'graph_learned_potential_{i}.pdf', format='pdf')
    plt.close()

    # eventually want to compare snapshot of evolution against evolution generated
    # from learned potential


    ###############################################################
    # propagate a0 with the learned potential
    ###############################################################

    # first recombine learned theta into vector of complex values
    # vtoeplearnedR = rsltadjthetarnd[:numtoepelms]
    # vtoeplearnedI = jnp.concatenate((jnp.array([0.0]), rsltadjthetarnd[numtoepelms:]))
    # vtoeplearned = vtoeplearnedR + 1j * vtoeplearnedI

    # construct vmatlearned from complex toeplitz vector
    # vmatlearned = jnp.concatenate([jnp.flipud(jnp.conj(vtoeplearned)), vtoeplearned[1:]])[toepindxmat]

    # Hamiltonian operator with true potential
    # in the Fourier representation
    # hmatlearned = kmat + vmatlearned

    # eigen-decomposition of the Hamiltonian matrix
    # spclearned, sttlearned = jnl.eigh(hmatlearned)

    # compute propagator matrix
    # propalearned = sttlearned @ jnp.diag(jnp.exp(-1j * spclearned * dt)) @ sttlearned.conj().T

    # propagate system starting from initial "a" state
    # using the Hamiltonian constructed from the true potential
    # (used for generating training data)
    # amatlearnedvec = []
    # for thisa0 in a0vec:
    #     tempamat = [thisa0.copy()]
    #     for i in range(numts):
    #         tempamat.append(propalearned @ tempamat[-1])
    #
    #     amatlearnedvec.append(tempamat)
    #
    # amatlearnedvec = jnp.array(amatlearnedvec)
    #
    # print('L2 error of amat:', nl.norm(amattruevec - amatlearnedvec, axis=0))

    # plot of real part of last state of system propagated with learned potential vs.
    # last state of amat
    # for i in range(len(amattruevec)):
    #     psiTlearned = amatlearnedvec[i, -1] @ fourtox
    #     psiTtrue = amattruevec[i, -1] @ fourtox
    #     plt.plot(xvec, jnp.real(psiTlearned), '.-', label='learned')
    #     plt.plot(xvec, jnp.real(psiTtrue), label='truth')
    #     plt.xlabel('x')
    #     plt.title('Real Part of Final State - Learned vs. Truth')
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig(cwddir / f'graph_real_part_last_state_learned_vs_truth_{i}.pdf', format='pdf')
    #     plt.close()