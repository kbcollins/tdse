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
# identify script on stdout
###############################################################

print('-------INVERSE-------')


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
# a0vec = np.load(workingdir / 'a0vec.npy')
amattruevec = np.load(cwddir / 'amattruevec.npy')

# fourtox = np.load(workingdir / 'fourtox.npy')
# vtruetoep = np.load(workingdir / 'vtruetoep.npy')
vxvec = np.load(cwddir / 'vtruexvec.npy')

print('Computational environment loaded.')
# print computational environment variables to stdout
print('L =', L)
print('numx =', numx)
print('numfour =', numfour)
print('numts =', numts)
print('dt =', dt)
print('Number of a0 states:', amattruevec.shape[0])


###############################################################
# recreate variables from loaded data
###############################################################

# real space grid points (for plotting)
xvec = np.linspace(-L, L, numx)

# vector of Fourier mode indices
# fournvec = -numfour,...,0,...,numfour
fournvec = np.arange(-numfour, numfour + 1)

# matrix for converting Fourier representation to real space
# used like realspacevec = fourspacevec @ fourtox
fourtox = np.exp(1j * np.pi * np.outer(fournvec, xvec) / L) / np.sqrt(2 * L)

# number of Toeplitz elements in the Fourier representation
numtoepelms = 2 * numfour + 1

# construct initial state vector
# a0vec = amattruevec[:, 0]
# print('Shape a0vec:', a0vec.shape)

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

# Toeplitz indexing matrix, used for constructing Toeplitz matrix
# from a vector setup like:
# jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
aa = (-1) * np.arange(0, numtoepelms).reshape(numtoepelms, 1)
bb = [np.arange(numtoepelms - 1, 2 * numtoepelms - 1)]
toepindxmat = np.array(aa + bb)
# print(toepindxmat.shape)


###############################################################
# theta
###############################################################

# true potential in the form of theta (for testing purposes)
# thetatrue = jnp.concatenate((jnp.real(vtruetoep), jnp.imag(vtruetoep[1:])))

# initialize theta with random coefficients close to zero
seed = 1234  # set to None for random initialization
print('seed =', seed)

thetarnd = 0.001 * np.random.default_rng(seed).normal(size=numtoepelms * 2 - 1)
thetarnd = jnp.array(thetarnd)

np.save(cwddir / 'thetarnd', thetarnd)
print('thetarnd saved.')


###############################################################
# define objective function
###############################################################

# define objective function
def ampsqobject(theta):
    ###############################################################
    # theta is a vector containing the concatenation
    # of the real and imaginary parts of vmat
    # its size should be 2 * numtoepelms - 1 = 4 * numfour + 1
    ###############################################################

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

    # forward propagation loop
    rtnobj = 0.0
    # for r in range(len(a0vec)):
    for r in range(betamatvec.shape[0]):
        # thisahat = a0vec[r].copy()
        thisahat = amattruevec[r, 0].copy()
        thisbetahatmat = [jnp.correlate(thisahat, thisahat, 'same') / jnp.sqrt(2 * L)]

        # propagate system starting from initial "a" state
        # for _ in range(numts):
        for _ in range(betamatvec.shape[1] - 1):
            # propagate the system one time-step
            thisahat = propahat @ thisahat
            # calculate the amp^2
            thisbetahatmat.append(jnp.correlate(thisahat, thisahat, 'same') / jnp.sqrt(2 * L))

        # compute objective functions
        tempresid = jnp.array(thisbetahatmat) - betamatvec[r]
        thisobj = 0.5 * jnp.sum(jnp.abs(tempresid)**2)
        rtnobj += thisobj

    return rtnobj

# jit ampsquaredobjective
jitampsqobject = jax.jit(ampsqobject)
# complie and test jitampsquaredobjective
# print(jitampsquaredobjective(thetatrue))


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
    # for r in range(len(a0vec)):
    for r in range(betamatvec.shape[0]):
        # propagate system starting from initial "a" state
        # thisahatmat = [a0vec[r].copy()]
        thisahatmat = [amattruevec[r, 0].copy()]
        thisbetahatmat = [jnp.correlate(thisahatmat[0], thisahatmat[0], 'same') / jnp.sqrt(2 * L)]
        # thisrhomat = [jnp.correlate(thisahatmat[0], thisahatmat[0], 'same') / jnp.sqrt(2 * L)]
        thispartlammat = [jnp.zeros(numtoepelms, dtype=complex)]

        # propagate system starting from thisa0vec state
        # for i in range(numts):
        for i in range(betamatvec.shape[1] - 1):
            # propagate the system one time-step and store the result
            thisahatmat.append(propahat @ thisahatmat[-1])

            # calculate the amp^2
            thisbetahatmat.append(jnp.correlate(thisahatmat[-1], thisahatmat[-1], 'same') / jnp.sqrt(2 * L))
            # thisrhomat.append(jnp.correlate(thisahatmat[-1], thisahatmat[-1], 'same') / jnp.sqrt(2 * L))

            # compute \rho^r - \beta^r
            # compute \betahat^r - \beta^r
            thiserr = thisbetahatmat[-1] - betamatvec[r, i+1]
            # thiserr = thisrhomat[-1] - betamatvec[r, i+1]

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


# jist adjgrads
jitadjgrads = jax.jit(adjgrads)
# compile and test jitadjgrads
# print(nl.norm(jitadjgrads(thetatrue)))


###############################################################
# learn theta
###############################################################

# start optimization (i.e., learning theta)
rsltadjthetarnd = so.minimize(fun=jitampsqobject, x0=thetarnd, jac=jitadjgrads, tol=1e-12, options={'maxiter': 4000, 'disp': True, 'gtol': 1e-15}).x
# thetahat = so.minimize(jitampsquaredobjective, thetarnd, jac=jitadjgrads, tol=1e-12, options={'maxiter': 1000, 'disp': True, 'gtol': 1e-15}).x

np.save(cwddir / 'thetahat', rsltadjthetarnd)
print('thetahat saved.')


###############################################################
# function for transforming theta to a real space potential
###############################################################

def thetatoreal(theta):
    thetaR = theta[:numtoepelms]
    thetaI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
    thetacomplex = thetaR + 1j * thetaI
    potentialfourier = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(thetacomplex[1:(numfour + 1)])), thetacomplex[:(numfour + 1)]])
    potentialreal = potentialfourier @ fourtox
    return potentialreal

# transform init theta to real space potentials
vinitrec = thetatoreal(thetarnd)

# transform learned theta to real space potential
vlearnrec = thetatoreal(rsltadjthetarnd)


###############################################################
# results
###############################################################

# learned potential vs initial potential
plt.plot(xvec, jnp.real(vlearnrec), '.-', label='Learned')
plt.plot(xvec, jnp.real(vinitrec), label='Initial')
plt.xlabel('x')
plt.title('Learned vs. Initial Potentials')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_inverse_learned_vs_initial_potential.pdf', format='pdf')
plt.close()

# learned potential vs true potential
plt.plot(xvec, jnp.real(vlearnrec), '.-', label='Learned')
plt.plot(xvec, vxvec, label='True')
plt.xlabel('x')
plt.title('Learned vs. True Potentials')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_inverse_true_vs_learned_potential.pdf', format='pdf')
plt.close()

# shifted learned potential vs true potential
midpointindex = numx // 2
print('midpointindex =', midpointindex)
shift = vxvec[midpointindex] - jnp.real(vlearnrec)[midpointindex]

# set trim to L=10
trim = np.where(xvec >= -10)[0][0]  # 125
print('trim =', trim)

# calculate and return l2 error
print('l2 error of learned potential:', nl.norm(jnp.real(vlearnrec) - vxvec), sep='\n')
print('l2 error of shifted learned potential:', nl.norm(jnp.real(vlearnrec) + shift - vxvec), sep='\n')
l2errshifttrim = nl.norm(jnp.real(vlearnrec)[trim:-trim] + shift - vxvec[trim:-trim])
print('l2 error of shifted and trimmed learned potential:', l2errshifttrim, sep='\n')

# calculate and return l2 error
print('l-inf error of learned potential:', np.amax(np.abs(jnp.real(vlearnrec) - vxvec)), sep='\n')
print('l-inf error of shifted learned potential:', np.amax(np.abs(jnp.real(vlearnrec) + shift - vxvec)), sep='\n')
linferrshifttrim = np.amax(np.abs(jnp.real(vlearnrec)[trim:-trim] + shift - vxvec[trim:-trim]))
print('l-inf error of shifted and trimmed learned potential:', linferrshifttrim, sep='\n')

# plot shifted potential
plt.plot(xvec, jnp.real(vlearnrec) + shift, '.-', label='Learned')
plt.plot(xvec, vxvec, label='True')
plt.xlabel('x')
plt.title(f'Shifted Learned Potential vs. True Potential\nl2 error (shift/trim) = {l2errshifttrim}\nl-inf error (shift/trim) = linferrshifttrim')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_inverse_shifted_true_vs_learned_potential.pdf', format='pdf')
plt.close()