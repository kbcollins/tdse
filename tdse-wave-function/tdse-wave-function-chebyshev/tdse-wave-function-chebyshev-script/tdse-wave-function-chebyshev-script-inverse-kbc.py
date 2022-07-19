import sys
import pathlib
import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import scipy.special as ss
import scipy.optimize as so
import scipy.integrate as si
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
print('')  # blank line


###############################################################
# set directories to load from and save to
###############################################################

# get path to directory containing amat from command line
cmdlinearg = sys.argv[1]
print('Command line argument:', cmdlinearg)

# transform commandline argument to path object
workingdir = pathlib.Path(cmdlinearg)
print('Current working directory:', workingdir)

# set identifier for saved output
savename = 'inverse'

# set directory to store results
resultsdir = workingdir / f'results-{savename}'
print('Results directory:', resultsdir)


###############################################################
# load computational parameters
###############################################################

# load saved computational parameters
cmpprm = np.load(workingdir / 'cmpprm.npy', allow_pickle=True)
print('cmpprm =', cmpprm)

# store loaded parameters as variables
L = float(cmpprm[0])
numx = int(cmpprm[1])
numfour = int(cmpprm[2])
dt = float(cmpprm[3])
numts = int(cmpprm[4])

# load state variables
a0vec = np.load(workingdir / 'a0vec.npy')
# propatrue = np.load(savedir / 'propatrue.npy')
amattruevec = np.load(workingdir / 'amattruevec.npy')

# load true potential
# vtruetoep = np.load(savedir / 'vtruetoep.npy')
vtruexvec = np.load(workingdir / 'vtruexvec.npy')

print('Computational variables loaded.')

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

# make kinetic operator in the Fourier representation
# (this is constant for a given system)
kmat = np.diag(np.arange(-numfour, numfour + 1) ** 2 * np.pi ** 2 / (2 * L ** 2))


###############################################################
# Set trim of real space region
###############################################################

trim = np.where(xvec >= -10)[0][0]  # 125
print('trim =', trim)
print('')  # blank line


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
# Chebyshev model
# - function to transform the model to real space
# - function to transform the model into Toeplitz vmat form
###############################################################

# set the number of Chebyshev coefficients
# the total number of Chebyshev coefficients = numcheb + 1
# from experience, odd values work best
numcheb = 11  # 41

kvec = np.arange(1, numcheb + 2)
chebnvec = np.arange(0, numcheb + 1)

# matrix for transforming Chebyshev representations to
# real space representation
# used like: chebtox @ cheb_cff_vec
chebtox = ss.eval_chebyt(np.expand_dims(chebnvec, 0), np.expand_dims(xvec / L, 1))

# matrix for transforming the Chebyshev representation
# to Fourier representation (this is used in the adjoint
# method to construct vhatmat)
chebtofour = []
for thischebn in range(numcheb + 1):
    temptoeprow = []
    for thisfourn in range(2 * numfour + 1):
        def intgrnd(x):
            return ss.eval_chebyt(thischebn, x / L) * np.exp(-1j * np.pi * thisfourn * x / L) / (2 * L)
        def rintgrnd(x):
            return intgrnd(x).real
        def iintgrnd(x):
            return intgrnd(x).imag
        temptoeprow.append(si.quad(rintgrnd, -L, L, limit=100)[0] + 1j * si.quad(iintgrnd, -L, L, limit=100)[0])
    chebtofour.append(sl.toeplitz(r=temptoeprow, c=np.conj(temptoeprow)))

# used like: chebtofour @ cheb_cff_vec
chebtofour = jnp.array(np.transpose(np.array(chebtofour), [1, 2, 0]))


###############################################################
# function for computing Chebyshev coefficients
# for a given function
###############################################################

def fntocheb(fn):
    chebweights = np.ones(numcheb + 1)
    chebweights[0] = 0.5

    def theta(k):
        return (k - 0.5) * np.pi / (numcheb + 1)

    def g(k):
        return fn(L * np.cos(theta(k)))

    chebvec = 2 * np.sum(g(kvec) * np.cos(chebnvec[..., np.newaxis] * theta(kvec)), axis=1) / (numcheb + 1)

    chebvec = chebweights * chebvec

    return chebvec


###############################################################
# theta
###############################################################

# initialize theta with random values
seed = 1234  # set to None for random initialization
print('seed =', seed)
thetarnd = 10.0 * np.random.default_rng(seed).uniform(size=numcheb + 1) - 5.0  # interval=[-5.0, 5.0)
# thetarnd = 0.02 * np.random.default_rng(seed).random(size=numcheb + 1) - 0.01  # interval=[-0.01, 0.01)
# thetarnd = 0.001 * np.random.default_rng(seed).normal(size=numcheb + 1)  # mean=0, std=1
thetarnd = jnp.array(thetarnd)

np.save(workingdir / 'thetarnd', thetarnd)
print('thetarnd saved.')


###############################################################
# objective function
###############################################################

def chebwaveobject(theta):
    #################################################
    # theta is a vector containing the Chebyshev
    # coefficients (i.e., the Chebyshev representation
    # of the potential)
    #################################################

    # construct the vmat from theta
    vhatmat = chebtofour @ theta

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # **************************************************
    # Construct Hamiltonian matrix
    hhatmat = kmat + vhatmat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hhatmat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T

    # forward propagation loop
    ahatmatvec = []
    for r in range(a0vec.shape[0]):
        thisahatmat = [a0vec[r].copy()]

        # propagate system starting from initial "a" state
        for _ in range(numts):
            # propagate the system one time-step
            thisahatmat.append(propahat @ thisahatmat[-1])

        ahatmatvec.append(thisahatmat)

    # transform python list to a jax.numpy array
    ahatmatvec = jnp.array(ahatmatvec)

    # compute objective functions
    resid = ahatmatvec - amattruevec

    # as per our math, we need to take the real part of
    # the objective
    rtnobjvec = jnp.real(0.5 * jnp.sum(jnp.conj(resid) * resid, axis=1))
    rtnobj = jnp.sum(rtnobjvec)
    # **************************************************

    return rtnobj


# jit wavefnobject()
jitchebwaveobject = jax.jit(chebwaveobject)


###############################################################
# adjoint method for computing gradient
###############################################################

# function for computing gradients using adjoint method
def chebwavegradsadj(theta):
    #################################################
    # theta is a vector containing the Chebyshev
    # coefficients (i.e., the Chebyshev representation
    # of the potential)
    #################################################

    # construct the vmat from theta
    vhatmat = chebtofour @ theta

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # **************************************************
    # Construct Hamiltonian matrix
    hhatmat = kmat + vhatmat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hhatmat)

    # compute propagator matrices
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T
    proplam = jnp.transpose(jnp.conjugate(propahat))

    # forward propagation
    ahatmatvec = []
    lammatvec = []
    for r in range(a0vec.shape[0]):
        ####################################################
        # build ahatmat, i.e., forward propagate of the
        # system with theta starting from a given initial
        # state a0
        ####################################################

        # initialize thisahatmat with given a0 state
        thisahatmat = [a0vec[r].copy()]

        # forward propagate of the system with theta
        for i in range(numts):
            # propagate the system one time-step and store the result
            thisahatmat.append(propahat @ thisahatmat[-1])

        # store compute ahatmat
        ahatmatvec.append(jnp.array(thisahatmat))

        ####################################################
        # build lammat
        # \lambda_N = (\hat{a}_N - a_N)
        # \lambda_j = (\hat{a}_j - a_j) + [\nabla_a \phi_{\Delta t} (a_j; \theta)]^\dagger \lambda_{j+1}
        # [\nabla_a \phi_{\Delta t} (a_j; \theta)]^\dagger = [exp{-i H \Delta t}]^\dagger
        ####################################################

        # compute the error of ahatmatvec[r] WRT amattruevec[r]
        thisahatmaterr = ahatmatvec[r] - amattruevec[r]

        # initialize thislammat
        thislammat = [thisahatmaterr[-1]]

        # build lammat backwards then flip
        for i in range(2, numts + 2):
            thislammat.append(thisahatmaterr[-i] + proplam @ thislammat[-1])

        # flip thislammat, make into a jax array, and add to lammatvec
        lammatvec.append(jnp.flipud(jnp.array(thislammat)))

    # make lists into JAX array object
    ahatmatvec = jnp.array(ahatmatvec)
    lammatvec = jnp.array(lammatvec)
    # **************************************************


    ####################################################
    # the remainder of this function is for computing
    # the gradient of the exponential matrix
    ####################################################

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # **************************************************
    offdiagmask = jnp.ones((numtoepelms, numtoepelms)) - jnp.eye(numtoepelms)
    expspec = jnp.exp(-1j * dt * spchat)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(spchat, spchat)
    denom = offdiagmask * (-1j * dt) * (s1 - s2) + jnp.eye(numtoepelms)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
    # **************************************************

    derivamat = jnp.einsum('ij,jkm,kl->ilm', jnp.transpose(jnp.conj(stthat)), chebtofour, stthat) * jnp.expand_dims(mask, 2)

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # **************************************************
    # because the Chebyshev coefficients are real valued
    # alldmat is half the size it would be if the coefficients
    # were complex (like with the Fourier basis)
    alldmat = -1j * dt * jnp.einsum('ij,jkm,kl->mil', stthat, derivamat, stthat.conj().T)

    # compute all entries of the gradient at once
    gradients = jnp.real(jnp.einsum('bij,ajk,bik->a', jnp.conj(lammatvec[:, 1:]), alldmat, ahatmatvec[:, :-1]))
    # **************************************************

    return gradients


# jist adjgrads
jitchebwavegradsadj = jax.jit(chebwavegradsadj)

# print(nl.norm(jitchebwaveadjgrads(thetatrue)))


###############################################################
# learning
###############################################################

# start optimization (i.e., learning theta)
thetahat = so.minimize(fun=jitchebwaveobject,
                       x0=thetarnd,
                       jac=jitchebwavegradsadj,
                       tol=1e-12,
                       options={'maxiter': 4000, 'disp': True, 'gtol': 1e-15}).x

np.save(workingdir / 'thetahat', thetahat)
print('thetahat saved.')


###############################################################
# results
###############################################################

# transform randtheta theta to real space potentials
vinitrec = chebtox @ thetarnd

# transform learned theta to real space potential
vlearnrec = chebtox @ thetahat

# learned potential vs initial potential
plt.plot(xvec, jnp.real(vlearnrec), '.-', label='Learned')
plt.plot(xvec, jnp.real(vinitrec), label='Initial')
plt.xlabel('x')
plt.title('Learned vs. Initial Potentials')
plt.legend()
# plt.show()
plt.savefig(resultsdir / f'graph_{savename}_learned_vs_initial_potential.pdf', format='pdf')
plt.close()

# learned potential vs true potential
plt.plot(xvec, jnp.real(vlearnrec), '.-', label='Learned')
plt.plot(xvec, vtruexvec, label='True')
plt.xlabel('x')
plt.title('Learned vs. True Potentials')
plt.legend()
# plt.show()
plt.savefig(resultsdir / f'graph_{savename}_true_vs_learned_potential.pdf', format='pdf')
plt.close()

# shifted learned potential vs true potential
midpointindex = numx // 2
print('midpointindex =', midpointindex)
shift = vtruexvec[midpointindex] - jnp.real(vlearnrec)[midpointindex]

# calculate and return l2 error
print('l2 error of learned potential:', nl.norm(jnp.real(vlearnrec) - vtruexvec), sep='\n')
print('l2 error of shifted learned potential:', nl.norm(jnp.real(vlearnrec) + shift - vtruexvec), sep='\n')
l2errshifttrim = nl.norm(jnp.real(vlearnrec)[trim:-trim] + shift - vtruexvec[trim:-trim])
print('l2 error of shifted and trimmed learned potential:', l2errshifttrim, sep='\n')

# calculate and return l2 error
print('l-inf error of learned potential:', np.amax(np.abs(jnp.real(vlearnrec) - vtruexvec)), sep='\n')
print('l-inf error of shifted learned potential:', np.amax(np.abs(jnp.real(vlearnrec) + shift - vtruexvec)), sep='\n')
linferrshifttrim = np.amax(np.abs(jnp.real(vlearnrec)[trim:-trim] + shift - vtruexvec[trim:-trim]))
print('l-inf error of shifted and trimmed learned potential:', linferrshifttrim, sep='\n')

# plot shifted potential
plt.plot(xvec, jnp.real(vlearnrec) + shift, '.-', label='Learned')
plt.plot(xvec, vtruexvec, label='True')
plt.xlabel('x')
plt.title(f'Shifted Learned Potential vs. True Potential\nl2 error (shift/trim) = {l2errshifttrim}\nl-inf error (shift/trim) = {linferrshifttrim}')
plt.legend()
# plt.show()
plt.savefig(resultsdir / f'graph_{savename}_shifted_true_vs_learned_potential.pdf', format='pdf')
plt.close()

print('')  # blank line