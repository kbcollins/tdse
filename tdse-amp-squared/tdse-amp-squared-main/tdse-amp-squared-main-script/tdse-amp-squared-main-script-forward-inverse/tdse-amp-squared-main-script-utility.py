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
# The scripts like tdse-amp-squared-main-script-forward-kbc.py
# generate training data, tests, and display results, the class
# handles the learning process. So once training has taken place
# the class object represents the best guess of the potential in
# the form of theta (i.e., thetahat)
#
# Class ideas
# - Class name: tdseadj
# - The object created by the class is theta
# - Instantiating theta
#   - has form thetahat = tdseadj(L, numx, numfour, numts, dt, seed=None)?
#   - theta is created and initialized with random values
#   - kmat is created
#   - toepindxmat
# - Methods
#   - initialize(numfour, seed=None): replaces all of theta's
#     values with random values
#   - thetatoreal(): transforms theta to real valued potential
#   - thetatovmat(): transforms theta to vmat form, ready to be
#     used by the adjoint method
#       - To transform theta to vmat, toepindxmat will have to
#         be constructed when theta is instantiated
#   - train(a0vec, betamatvec): given the training data a0vec and
#     betamatvec, this class method uses the adjoint method to
#     learn the true potential
#   - propagate(a0vec, numts, dt): propagates all states in a0vec
#     numts steps
#       - numts and dt here are different then the internal numts
#         and dt, the internal variables are about the training
#         data a0vec and betamatvec
###############################################################


###############################################################
# Toeplitz indexing matrix
###############################################################

# this doesn't really need to be a function, just drop the code
# in the instantiation method for the class object and store the
# result internally as _toepindxmat

def mk_toepindxmat(numtoepelms):
    # Toeplitz indexing matrix, used for constructing Toeplitz matrix
    # from a vector setup like:
    # jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
    aa = (-1) * np.arange(0, numtoepelms).reshape(numtoepelms, 1)
    bb = [np.arange(numtoepelms - 1, 2 * numtoepelms - 1)]
    toepindxmat = np.array(aa + bb)
    # print(toepindxmat.shape)

    return toepindxmat


###############################################################
# Internal function for transforming theta to flattened
# Toeplitz form
###############################################################

def _thetatotoep(theta, numfour):
    # internal method, puts theta into the flattened Toeplitz form
    # this is what can be changed depending on the model used for theta

    #################################################
    # theta is a vector containing the concatenation
    # of the real and imaginary parts of vmat
    # its size should be
    # 2 * numtoepelms - 1 = 4 * numfour + 1
    #################################################

    numtoepelms = 4 * numfour + 1

    # to use theta we need to first recombine the real
    # and imaginary parts into a vector of complex values
    thetatoepR = theta[:numtoepelms]
    thetatoepI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
    thetatoepComplex = thetatoepR + 1j * thetatoepI

    return thetatoepComplex

###############################################################
# function for transforming theta to vmat
###############################################################

def thetatovmat(theta, numfour, toepindxmat):
    thetatoep = _thetatotoep(theta, numfour)

    # construct vmathat from complex toeplitz vector
    vmathat = jnp.concatenate([jnp.flipud(jnp.conj(thetatoep)), thetatoep[1:]])[toepindxmat]

    return vmathat


###############################################################
# function for transforming theta to a real space potential
###############################################################

def thetatofour(theta, numfour, L):
    thetatoep = _thetatotoep(theta, numfour)

    potentialfourier = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(thetatoep[1:(numfour + 1)])), thetatoep[:(numfour + 1)]])

    return potentialfourier


###############################################################
# This function constructs the Hamiltonian operator matrix
# and performing an eigen-decomposition.
# Returned is the eigen-spectrum and eigen-states
###############################################################

def _hmateigh(theta, numfour, toepindxmat, kmat):
    # internal method
    # the first argument is like self
    # numfour, toepindxmat, and kmat are internal to the object

    vmathat = thetatovmat(theta, numfour, toepindxmat)

    # Construct Hamiltonian matrix
    hmathat = kmat + vmathat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hmathat)

    return (spchat, stthat)


###############################################################
# function for computing the forward propagation
###############################################################

def propagate(theta, a0vec, numts, dt, L, numfour, toepindxmat, kmat):
    # the first argument is like self
    # a0vec, numts, and dt are the arguments passed to the method
    # L, numfour, toepindxmat, and kmat are internal to the object
    # method returns betamathatvec and ahatmat

    # eigen-decomposition of the Hamiltonian operator matrix
    spchat, stthat = _hmateigh(theta, numfour, toepindxmat, kmat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T

    # forward propagation loop
    betamathatvec = []
    ahatmatvec = []
    for r in range(len(a0vec)):
    # for r in range(betamatvec.shape[0]):
        thisa0vec = a0vec[r].copy()
        thisahatmat = [thisa0vec]
        thisbetahatmat = [jnp.correlate(thisa0vec, thisa0vec, 'same') / jnp.sqrt(2 * L)]

        # propagate system starting from initial "a" state
        for _ in range(numts):
        # for _ in range(betamatvec.shape[1] - 1):
            # propagate the system one time-step
            thisahatmat.append(propahat @ thisahatmat[-1])
            # calculate the amp^2
            thisbetahatmat.append(jnp.correlate(thisahatmat[-1], thisahatmat[-1], 'same') / jnp.sqrt(2 * L))

        betamathatvec.append(thisbetahatmat)
        ahatmatvec.append(thisahatmat)

    return (jnp.array(betamathatvec), jnp.array(ahatmatvec))


###############################################################
# objective function
###############################################################

def ampsqobject(theta, a0vec, betamatvec, numts, dt, L, numfour, toepindxmat, kmat):
    betahatmatvec, _ = propagate(theta, a0vec, numts, dt, L, numfour, toepindxmat, kmat)

    # compute objective function for each a0 all at once
    resid = betahatmatvec - betamatvec
    objvec = 0.5 * jnp.sum(jnp.abs(resid)**2, axis=1)
    print(objvec.shape)  # make sure the same is equal to the number of a0 states

    return jnp.sum(objvec)

# jit ampsquaredobjective
jitampsqobject = jax.jit(ampsqobject)
# complie and test jitampsquaredobjective
# print(jitampsquaredobjective(thetatrue))


###############################################################
# adjoint method for computing gradient
###############################################################

# function for generating M and P matrix (used in adjoint method)
def mk_M_and_P(avec):
    # this could probably be transformed to an internal class method

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
def adjgrads(theta, a0vec, betamatvec, numts, dt, L, numfour, toepindxmat, kmat):
    # the first argument is like self
    # a0vec, numts, and dt are the arguments passed to the method
    # L, numfour, toepindxmat, and kmat are internal to the object

    numtoepelms = 2 * numfour + 1

    # eigen-decomposition of the Hamiltonian operator matrix
    spchat, stthat = _hmateigh(theta, numfour, toepindxmat, kmat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T
    proplam = jnp.transpose(jnp.conjugate(propahat))

    betahatmatvec, ahatmatvec = propagate(theta, a0vec, numts, dt, L, numfour, toepindxmat, kmat)
    betahatmaterrvec = betahatmatvec[1:] - betamatvec[1:]

    lammatvec = []
    for r in range(len(a0vec)):
        thispartlammat = [jnp.zeros(numtoepelms, dtype=complex)]

        # lambda propagation
        for i in range(1, numts + 1):
            # compute M and P matrix for lambda mat
            thisMmat, thisPmat = jit_mk_M_and_P(ahatmatvec[r, i])

            # compute part of lambda mat
            # ( 1 / \sqrt{2 L} ) * [ ( M^r )^\dagger * ( \rho^r - \beta^r )
            # + \overline{( P^r )^\dagger * ( \rho^r - \beta^r )} ]
            thispartlammat.append((thisMmat.conj().T @ betahatmaterrvec[r, i] + (thisPmat.conj().T @ betahatmaterrvec[r, i]).conj()) / jnp.sqrt(2 * L))

        # build lammat backwards then flip at the end
        thislammat = [thispartlammat[-1]]
        for i in range(2, numts + 2):
            thislammat.append(thispartlammat[-i] + proplam @ thislammat[-1])

        lammatvec.append(jnp.flipud(jnp.array(thislammat)))

    # make lists into JAX array object
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
# print(nl.norm(jitadjgrads(thetatrue)))

