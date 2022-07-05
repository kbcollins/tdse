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

print('-------CONVERGE TEST-------')


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
a0vec = np.load(cwddir / 'a0vec.npy')
propatrue = np.load(cwddir / 'propatrue.npy')
amattruevec = np.load(cwddir / 'amattruevec.npy')

# fourtox = np.load(cwddir / 'fourtox.npy')
# vtoeptrue = np.load(cwddir / 'vtoeptrue.npy')
vxvec = np.load(cwddir / 'vxvec.npy')
vxrange = np.amax(vxvec) - np.amin(vxvec)

print('Computational environment loaded.')
# print computational environment variables to stdout
print('L =', L)
print('numx =', numx)
print('numfour =', numfour)
print('numts =', numts)
print('dt =', dt)
print('Number of a0 states:', a0vec.shape[0])
print('')  # blank line


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
# define objective function
###############################################################

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
# function for transforming theta to a real space potential
###############################################################

def thetatoreal(theta):
    thetaR = theta[:numtoepelms]
    thetaI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
    thetacomplex = thetaR + 1j * thetaI
    potentialfourier = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(thetacomplex[1:(numfour + 1)])), thetacomplex[:(numfour + 1)]])
    potentialreal = potentialfourier @ fourtox
    return potentialreal


###############################################################
# Function for computing the propagator matrix given some theta
#   - theta is a vector containing the concatenation
#     of the real and imaginary parts of vmat
#   - theta should contain 2 * numtoepelms - 1
#     = 4 * numfour + 1 elements
###############################################################

def thetatopropmat(theta):
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

    return propahat


###############################################################
# Convergence test
#   Initialize theta and learn the true potential 200 times.
#   For each iteration, store the l2 and l-inf errors.
#   - Save plot of the errors.
#   - Compute the average of errors.
#   - Compute the distribution of the errors around their mean.
###############################################################

# set number of trials
numtrials = 4  # 200

# find midpoint of computational interval
midpointindex = numx // 2
print('midpointindex =', midpointindex)
trim = np.where(xvec >= -10)[0][0]  # 125
print('trim =', trim)

rawl2err = []
rawlinferr = []
shiftl2err = []
shiftlinferr = []
trimshiftl2err = []
trimshiftlinferr = []
for i in range(numtrials):
    if i % 10 == 0:
        print(f'{i} of {numtrials}')

    # initialize theta with random coefficients close to zero
    seed = None  # set to None for random initialization
    thetarnd = 0.02 * np.random.default_rng(seed).random(size=numtoepelms * 2 - 1) - 0.01  # interval [-0.01, 0.01)
    # thetarnd = 0.001 * np.random.default_rng(seed).normal(size=numtoepelms * 2 - 1)
    thetarnd = jnp.array(thetarnd)

    # start optimizing (i.e., learning)
    thisresult = so.minimize(fun=jitampsqobject,
                             x0=thetarnd, jac=jitadjgrads,
                             tol=1e-12, options={'maxiter': 1000, 'disp': False, 'gtol': 1e-15}).x
                             # tol=1e-12, options={'maxiter': 4000, 'disp': True, 'gtol': 1e-15}).x

    #################################################
    # propagate a0vec using the learned potential and
    # compare ahatmat to amat
    #   - these results are closer to what is
    #     available for real world problems, but
    #     would permit potentials which are not equal
    #     to the true potential
    #################################################

    # compute propagator from thisresult
    thispropahat = thetatopropmat(thisresult)

    # propagate starting from a0vec states
    amattruevec = []
    ahatmatvec = []
    for thisa0 in a0vec:
        tempamat = [thisa0.copy()]
        tempahatmat = [thisa0.copy()]
        for _ in range(numts):
            tempamat.append(propatrue @ tempamat[-1])
            tempahatmat.append(thispropahat @ tempahatmat[-1])

        amattruevec.append(tempamat)
        ahatmatvec.append(tempahatmat)

    # amattruevec = jnp.array(amattruevec)
    # ahatmatvec = jnp.array(ahatmatvec)

    ahatmatl2err = nl.norm(np.subtract(amattruevec, ahatmatvec))

    if i == 0:
        print('initializing ahatmatl2errbest and thetabestprop')
        ahatmatl2errbest = ahatmatl2err
        thetabestprop = thisresult
    else:
        if ahatmatl2errbest > ahatmatl2err:
            ahatmatl2errbest = ahatmatl2err
            thetabestprop = thisresult


    #################################################
    # learned potential vs. true potential
    #   - these results do a better job evaluate how
    #     well the method learns the true potential,
    #     but it is unrealistic because if you already
    #     know the true potential, you don't need a
    #     method to find it
    #################################################

    # get real space potential from learned theta
    thisvlearnrec = thetatoreal(thisresult)

    rawl2err.append(nl.norm(jnp.real(thisvlearnrec) - vxvec))
    rawlinferr.append(np.amax(np.abs(jnp.real(thisvlearnrec) - vxvec)))

    shift = vxvec[midpointindex] - jnp.real(thisvlearnrec)[midpointindex]
    shiftl2err.append(nl.norm(jnp.real(thisvlearnrec) + shift - vxvec))
    shiftlinferr.append(np.amax(np.abs(jnp.real(thisvlearnrec) + shift - vxvec)))

    trimshiftl2err.append(nl.norm(jnp.real(thisvlearnrec)[trim:-trim] + shift - vxvec[trim:-trim]))
    trimshiftlinferr.append(np.amax(np.abs(jnp.real(thisvlearnrec)[trim:-trim] + shift - vxvec[trim:-trim])))

    if i == 0:
        print('initializing trimshiftl2errbest and thetabestv')
        trimshiftl2errbest = trimshiftl2err[-1]
        thetabestv = thisresult
    else:
        if trimshiftl2errbest > trimshiftl2err[-1]:
            trimshiftl2errbest = trimshiftl2err[-1]
            thetabestv = thisresult


###############################################################
# Save theta which produces the potential with the lowest
# l2 error, thetabestv, and theta which produces the propagation
# which has the lowest l2 error W.R.T. the training data,
# thetabestprop
###############################################################

np.save(cwddir / 'thetabestprop', thetabestprop)
print('thetabestprop saved.')

np.save(cwddir / 'thetabestv', thetabestv)
print('thetabestv saved.')

print('')  # blank line

print('l2 error of thetabestprop W.R.T. thetabestv:', nl.norm(thetabestprop - thetabestv), sep='\n')

print('')  # blank line


###############################################################
# results
###############################################################

# plot rawl2err, shiftl2err, and trimshiftl2err together
plt.plot(rawl2err, '.', label='rawl2err')
plt.plot(shiftl2err, '.', label='shiftl2err')
plt.plot(trimshiftl2err, '.', label='trimshiftl2err')
plt.title(f'l2 Error of the Learned Potentials - {numtrials} Initializations')
plt.xlabel('Trial Number')
plt.ylabel('Error')
plt.legend()
plt.savefig(cwddir / 'graph_converge-test_l2_error_all.pdf', format='pdf')
plt.close()


# plot rawlinferr, shiftlinferr, and trimshiftlinferr together
plt.plot(rawlinferr, '.', label='rawlinferr')
plt.plot(shiftlinferr, '.', label='shiftlinferr')
plt.plot(trimshiftlinferr, '.', label='trimshiftlinferr')
plt.title(f'l-infinite Error of the Learned Potentials - {numtrials} Initializations')
plt.xlabel('Trial Number')
plt.ylabel('Error')
plt.legend()
plt.savefig(cwddir / 'graph_converge-test_l-infinite_error_all.pdf', format='pdf')
plt.close()


# results of raw potential l2 error
meanrawl2err = np.mean(rawl2err)
minrawl2err = np.amin(rawl2err)
maxrawl2err = np.amax(rawl2err)
avgdevrawl2err = np.mean(np.abs(np.subtract(rawl2err, meanrawl2err)))
prcnterrrawl2err = meanrawl2err / vxrange * 100
print('Mean of rawl2err:', meanrawl2err)
print('Minumum of rawl2err:', minrawl2err)
print('Maximum of rawl2err:', maxrawl2err)
print('Average deviation of rawl2err:', avgdevrawl2err)
print('Average % error of raw learned potential:', prcnterrrawl2err, sep='\n')

# plot raw potential l2 error
plt.plot(rawl2err, '.', label='rawl2err')
plt.hlines(meanrawl2err, xvec[0], xvec[-1], label='Mean')
plt.hlines(meanrawl2err + avgdevrawl2err, xvec[0], xvec[-1], '-.', label='_')
plt.hlines(meanrawl2err - avgdevrawl2err, xvec[0], xvec[-1], '-.', label='_')
plt.title(f'l2 Error of the Raw Learned Potentials - {numtrials} Trials\nmin={minrawl2err:.4}, max={maxrawl2err:.4},% error = {prcnterrrawl2err:.4}')
plt.xlabel('Trial')
plt.ylabel('Error')
plt.legend()
plt.savefig(cwddir / 'graph_converge-test_l2_error_rawl2err.pdf', format='pdf')
plt.close()

# results of raw potential l-infinite error
meanrawlinferr = np.mean(rawlinferr)
minrawlinferr = np.amin(rawlinferr)
maxrawlinferr = np.amax(rawlinferr)
avgdevrawlinferr = np.mean(np.abs(np.subtract(rawlinferr, meanrawlinferr)))
prcnterrrawlinferr = meanrawlinferr / vxrange * 100
print('Mean of rawlinferr:', meanrawlinferr)
print('Minumum of rawlinferr:', minrawlinferr)
print('Maximum of rawlinferr:', maxrawlinferr)
print('Average deviation of rawlinferr:', avgdevrawlinferr)
print('Max % error of raw learned potential:', prcnterrrawlinferr, sep='\n')

# plot raw potential l-infinite error
plt.plot(rawlinferr, '.', label='rawlinferr')
plt.hlines(meanrawlinferr, xvec[0], xvec[-1], label='Mean')
plt.hlines(meanrawlinferr + avgdevrawlinferr, xvec[0], xvec[-1], '-.', label='_')
plt.hlines(meanrawlinferr - avgdevrawlinferr, xvec[0], xvec[-1], '-.', label='_')
plt.title(f'l-infinite Error of the Raw Learned Potentials - {numtrials} Trials\nmin={minrawlinferr:.4}, max={maxrawlinferr:.4},% error = {prcnterrrawlinferr:.4}')
plt.xlabel('Trial Number')
plt.ylabel('Error')
plt.legend()
plt.savefig(cwddir / 'graph_converge-test_l-infinite_error_rawlinferr.pdf', format='pdf')
plt.close()


# results of shifted potential l2 error
meanshiftl2err = np.mean(shiftl2err)
minshiftl2err = np.amin(shiftl2err)
maxshiftl2err = np.amax(shiftl2err)
avgdevshiftl2err = np.mean(np.abs(np.subtract(shiftl2err, meanshiftl2err)))
prcnterrshiftl2err = meanshiftl2err / vxrange * 100
print('Mean of shiftl2err:', meanshiftl2err)
print('Minumum of shiftl2err:', minshiftl2err)
print('Maximum of shiftl2err:', maxshiftl2err)
print('Average deviation of shiftl2err:', avgdevshiftl2err)
print('Average % error of shifted learned potential:', prcnterrshiftl2err, sep='\n')

# plot shifted potential l2 error
plt.plot(shiftl2err, '.', label='shiftl2err')
plt.hlines(meanshiftl2err, xvec[0], xvec[-1], label='Mean')
plt.hlines(meanshiftl2err + avgdevshiftl2err, xvec[0], xvec[-1], '-.', label='_')
plt.hlines(meanshiftl2err - avgdevshiftl2err, xvec[0], xvec[-1], '-.', label='_')
plt.title(f'l2 Error of the Shifted Learned Potentials - {numtrials} Trials\nmin={minshiftl2err:.4}, max={maxshiftl2err:.4},% error = {prcnterrshiftl2err:.4}')
plt.xlabel('Trial Number')
plt.ylabel('Error')
plt.legend()
plt.savefig(cwddir / 'graph_converge-test_l2_error_shiftl2err.pdf', format='pdf')
plt.close()

# results of shifted potential l-infinite error
meanshiftlinferr = np.mean(shiftlinferr)
minshiftlinferr = np.amin(shiftlinferr)
maxshiftlinferr = np.amax(shiftlinferr)
avgdevshiftlinferr = np.mean(np.abs(np.subtract(shiftlinferr, meanshiftlinferr)))
prcnterrshiftlinferr = meanshiftlinferr / vxrange * 100
print('Mean of shiftlinferr:', meanshiftlinferr)
print('Minumum of shiftlinferr:', minshiftlinferr)
print('Maximum of shiftlinferr:', maxshiftlinferr)
print('Average deviation of shiftlinferr:', avgdevshiftlinferr)
print('Max % error of shifted learned potential:', prcnterrshiftlinferr, sep='\n')

# plot shifted potential l-infinite error
plt.plot(shiftlinferr, '.', label='shiftlinferr')
plt.hlines(meanshiftlinferr, xvec[0], xvec[-1], label='Mean')
plt.hlines(meanshiftlinferr + avgdevshiftlinferr, xvec[0], xvec[-1], '-.', label='_')
plt.hlines(meanshiftlinferr - avgdevshiftlinferr, xvec[0], xvec[-1], '-.', label='_')
plt.title(f'l-infinite Error of the Shifted Learned Potentials - {numtrials} Trials\nmin={minshiftlinferr:.4}, max={maxshiftlinferr:.4},% error = {prcnterrshiftlinferr:.4}')
plt.xlabel('Trial Number')
plt.ylabel('Error')
plt.legend()
plt.savefig(cwddir / 'graph_converge-test_l-infinite_error_shiftlinferr.pdf', format='pdf')
plt.close()


# results of trimmed and shifted potential l2 error
meantrimshiftl2err = np.mean(trimshiftl2err)
mintrimshiftl2err = np.amin(shiftlinferr)
maxtrimshiftl2err = np.amax(shiftlinferr)
avgdevtrimshiftl2err = np.mean(np.abs(np.subtract(shiftlinferr, meanshiftlinferr)))
prcnterrtrimshiftl2err = meanshiftlinferr / vxrange * 100
print('Mean of trimshiftl2err:', meantrimshiftl2err)
print('Minumum of trimshiftl2err:', mintrimshiftl2err)
print('Maximum of trimshiftl2err:', maxtrimshiftl2err)
print('Average deviation of trimshiftl2err:', meantrimshiftl2err)
print('Average % error of trimmed and shifted learned potential:', prcnterrtrimshiftl2err, sep='\n')

# plot trimmed and shifted potential l2 error
plt.plot(trimshiftl2err, '.', label='trimshiftl2err')
plt.hlines(meantrimshiftl2err, xvec[0], xvec[-1], label='Mean')
plt.hlines(meantrimshiftl2err + avgdevtrimshiftl2err, xvec[0], xvec[-1], '-.', label='_')
plt.hlines(meantrimshiftl2err - avgdevtrimshiftl2err, xvec[0], xvec[-1], '-.', label='_')
plt.title(f'l2 Error of Trimmed and Shifted\nLearned Potentials - {numtrials} Trials\nmin={mintrimshiftl2err:.4}, max={maxtrimshiftl2err:.4},% error = {prcnterrtrimshiftl2err:.4}')
plt.xlabel('Trial Number')
plt.ylabel('Error')
plt.legend()
plt.savefig(cwddir / 'graph_converge-test_l2_error_trimshiftl2err.pdf', format='pdf')
plt.close()

# results of trimmed and shifted potential l-infinite error
meantrimshiftlinferr = np.mean(trimshiftlinferr)
mintrimshiftlinferr = np.amin(shiftlinferr)
maxtrimshiftlinferr = np.amax(shiftlinferr)
avgdevtrimshiftlinferr = np.mean(np.abs(np.subtract(shiftlinferr, meanshiftlinferr)))
prcnterrtrimshiftlinferr = meanshiftlinferr / vxrange * 100
print('Mean of trimshiftlinferr:', meantrimshiftlinferr)
print('Minumum of trimshiftlinferr:', mintrimshiftlinferr)
print('Maximum of trimshiftlinferr:', maxtrimshiftlinferr)
print('Average deviation of trimshiftlinferr:', avgdevtrimshiftlinferr)
print('Max % error of trimmed and shifted learned potential:', prcnterrtrimshiftlinferr, sep='\n')

# plot trimmed and shifted potential l-infinite error
plt.plot(trimshiftlinferr, '.', label='trimshiftlinferr')
plt.hlines(meantrimshiftlinferr, xvec[0], xvec[-1], label='Mean')
plt.hlines(meantrimshiftlinferr + avgdevtrimshiftlinferr, xvec[0], xvec[-1], '-.', label='_')
plt.hlines(meantrimshiftlinferr - avgdevtrimshiftlinferr, xvec[0], xvec[-1], '-.', label='_')
plt.title(f'l-infinite Error of the Trimmed and Shifted\nLearned Potentials - {numtrials} Trials\nmin={mintrimshiftlinferr:.4}, max={maxtrimshiftlinferr:.4},% error = {prcnterrtrimshiftlinferr:.4}')
plt.xlabel('Trial Number')
plt.ylabel('Error')
plt.legend()
plt.savefig(cwddir / 'graph_converge-test_l-infinite_error_trimshiftlinferr.pdf', format='pdf')
plt.close()