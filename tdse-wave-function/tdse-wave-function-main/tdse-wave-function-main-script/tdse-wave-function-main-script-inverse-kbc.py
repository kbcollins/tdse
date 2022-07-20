import sys
import pathlib
from time import time_ns
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

import tdsemodelclass

# save initial time of script
timetotalstart = time_ns()


###############################################################
# identify script on stdout
###############################################################

scriptID = 'inverse'
print(f'-------{scriptID.upper()}-------')


###############################################################
# get commandline arguments
# - cmdlineargmodel: choice of model. Store as a string.
#   possible selections are {'fourier', 'cheby'}
# - cmdlineargsavedir: directory to load/save data to/from
###############################################################

print('sys.argv =', sys.argv)

# model of potentil
cmdlineargmodel = sys.argv[1]
print('cmdlineargmodel =', cmdlineargmodel)

# path to directory containing amat from command line
cmdlinearg = sys.argv[2]
print('Command line argument:', cmdlinearg)


###############################################################
# set path to working directory
###############################################################

# file path to directory where output (like training data)
# should be saved. The directory should be set up like:
# ./<parent>/<potential>/<trial>/
workdir = pathlib.Path(cmdlinearg)
print('Current working directory:', workdir)


###############################################################
# load computational parameters
###############################################################

# load saved computational parameters
cmpprm = np.load(workdir / 'cmpprm.npy', allow_pickle=True)
print('cmpprm =', cmpprm)

print('')  # blank line

# store loaded parameters as variables
L = float(cmpprm[0])
numx = int(cmpprm[1])
numfour = int(cmpprm[2])
dt = float(cmpprm[3])
numts = int(cmpprm[4])

# load state variables
a0vec = np.load(workdir / 'a0vec.npy')
# propatrue = np.load(workdir / 'propatrue.npy')
amattruevec = np.load(workdir / 'amattruevec.npy')

# load true potential
# vtruetoep = np.load(workdir / 'vtruetoep.npy')
vtruexvec = np.load(workdir / 'vtruexvec.npy')


###############################################################
# open agg-results-<scriptID> for writing
###############################################################

# Open data agg-results-<scriptID> files in append mode.
# These files collect important results from all the trials
# of one specific potential. Directroy like:
# ./"$parentdir"/v"$thisv"/"$thistrial"/f'agg-results-{scriptID}.txt'
aggresultstxt = open(workdir.parent / f'agg-results-{scriptID}.txt', mode='a')

# set helpful strings to be used when writing to
# aggresultstxt
delim=', '
newline='\n'


###############################################################
# output computational parameters used and add them to
# the agg results file
###############################################################

# print computational environment variables to stdout
print('L =', L)
aggresultstxt.write(str(L) + delim)

print('numx =', numx)
aggresultstxt.write(str(numx) + delim)

print('numfour =', numfour)
aggresultstxt.write(str(numfour) + delim)

print('numts =', numts)
aggresultstxt.write(str(numts) + delim)

print('dt =', dt)
aggresultstxt.write(str(dt) + delim)

print('Number of a0 states:', a0vec.shape[0])
aggresultstxt.write(str(a0vec.shape[0]) + delim)

print('Model:', cmdlineargmodel)
aggresultstxt.write(cmdlineargmodel + delim)


###############################################################
# Identify what model to use to approximate the potential and
# specify the parameters which fully define the model
# - modelprms is a tuple containing all of the variables the
#   model needs to be fully defined
# - the '*' in *modelprms unpacks modelprms then passes
#   everything as the parameters to the instantiation of a
#   model object
###############################################################

if cmdlineargmodel == 'fourier':
    # Fourier model
    modelprms = (L, numx, numfour)
    model = tdsemodelclass.fourier
    # set number of basis used for the model (when using the
    # Fourier model, we use the same number of basis
    # as was used to discretize)
    nummodel = numfour
elif cmdlineargmodel == 'cheby':
    # Chebyshev basis functions
    # - From experience, I have found that cheby
    #   works best when numcheb is an odd numbers
    numcheb = 61  # 11
    # set number of basis used for the model
    nummodel = numcheb
    modelprms = (L, numx, numfour, numcheb)
    model = tdsemodelclass.cheby
else:
    print(f'Model selection "{cmdlineargmodel}" not recognized.')

# write number of basis used for the model
aggresultstxt.write(str(nummodel) + delim)

print('')  # blank line


###############################################################
# set directory to save results
###############################################################

# results directory
# workdir = ./"$parentdir"/v"$thisv"/"$thistrial"
# want: ./"$parentdir"/v"$thisv"/"$thistrial"/"$thismodel"/results-"$SLURM_JOB_NAME"
resultsdir = workdir / cmdlineargmodel / f'results-{scriptID}'
print('Results directory:', resultsdir)


###############################################################
# utilities - created from the loaded computational parameters
###############################################################

# real space grid points (for plotting)
xvec = np.linspace(-L, L, numx)

# vector of Fourier mode indices
# fournvec = -numfour,...,0,...,numfour
fournvec = np.arange(-numfour, numfour + 1)

# matrix for converting Fourier representation to real space
# - this converts functions in terms of the Fourier basis,
#   i.e., fn(x) = \sum_{n=-F}^F c_n \phi_n(x)
# - this does not convert vmat to real
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
# Set trim of real space region
###############################################################

trim = np.where(xvec >= -10)[0][0]  # 125
print('trim =', trim)
print('')  # blank line


###############################################################
# objective function
###############################################################

def waveobject(theta):
    #################################################
    # theta is the data structure given to the
    # optimizer which contains the potential in terms
    # of the model
    # - to use JAX, theta must be a JAX recognized
    #   object
    #################################################

    # **************************************************
    # the following code enclosed by ' # ****' is the
    # same regardless of the model use
    # **************************************************
    # construct vmathat using the model class method
    # .tovmat(), theta is what ever the model class
    # uses as the data structure to store the potential
    # any other arguments are what is required to define
    # the model
    vhatmat = model.thetatovmat(theta, *modelprms)

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
jitwaveobject = jax.jit(waveobject)

# precomplie jitfourwaveobject
print('jitwaveobject(model.randtheta(*modelprms, seed=1234)):', jitwaveobject(model.randtheta(*modelprms, seed=1234)), sep='\n')


###############################################################
# gradient computed using the adjoint method
###############################################################

def wavegradsadj(theta):
    #################################################
    # theta is the data structure given to the
    # optimizer which contains the potential in terms
    # of the model
    # - to use JAX, theta must be a JAX recognized
    #   object
    #################################################

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # **************************************************
    # construct vmathat using the model class method
    # .tovmat(), theta is what ever the model class
    # uses as the data structure to store the potential
    # any other arguments are what is required to define
    # the model
    vhatmat = model.thetatovmat(theta, *modelprms)

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


    ####################################################
    # The remainder of this function is for computing
    # the gradient of the exponential matrix
    # - Given the diagonalization H = U D U^\dagger
    # - The final gradient \nabla_\theta \phi(a;\theta)
    #   is Q = U M U^\dagger, where M = A (*) mask
    #   and A = U^\dagger [\nabla_\theta H
    #         = \nabla_\theta model of vhatmat or v(x)] U
    ####################################################
    offdiagmask = jnp.ones((numtoepelms, numtoepelms)) - jnp.eye(numtoepelms)
    expspec = jnp.exp(-1j * dt * spchat)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(spchat, spchat)
    denom = offdiagmask * (-1j * dt) * (s1 - s2) + jnp.eye(numtoepelms)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)

    # get the gradient of the model, function expects
    # model.thetatograd(theta, *args), where args is
    # what ever elements the model needs to be fully defined
    modelgrad = model.thetatograd(theta, *modelprms)

    # this line computes U^\dagger \nabla_\theta H(\theta) U
    prederivamat = jnp.einsum('ij,jkm,kl->ilm', stthat.conj().T, modelgrad, stthat)

    # this line computes
    # M_{i l} = A_{i l} (exp(D_{i i}) or (exp(D_{i i}) - exp(D_{l l}))/(D_{i i} - D_{l l}))
    derivamat = prederivamat * jnp.expand_dims(mask, 2)

    # this line computes Q = U M U^\dagger
    alldmat = -1j * dt * jnp.einsum('ij,jkm,kl->mil', stthat, derivamat, stthat.conj().T)

    # compute all entries of the gradient at once
    gradients = jnp.real(jnp.einsum('bij,ajk,bik->a', jnp.conj(lammatvec[:, 1:]), alldmat, ahatmatvec[:, :-1]))
    # **************************************************

    return gradients


# jist adjgrads
jitwavegradsadj = jax.jit(wavegradsadj)

# precompile jitwavegradsadj
print('nl.norm(jitwavegradsadj(model.randtheta(*modelprms, seed=1234))):', nl.norm(jitwavegradsadj(model.randtheta(*modelprms, seed=1234))), sep='\n')

print('')  # blank line

###############################################################
# learning
###############################################################

# create a model object and initialize it with random values
thetahat = model(*modelprms, seed=1234)

# transform the initialized thetahat to real space potential
# and store
vinitrec = thetahat.tox()

# start timing how long it takes to optimize
timeoptstart = time_ns()

# start optimization (i.e., learning theta) and store
# learned result in thetahat
optresults = so.minimize(fun=jitwaveobject,
                         x0=thetahat.theta,
                         jac=jitwavegradsadj,
                         tol=1e-12,
                         options={'maxiter': 4000, 'gtol': 1e-15})
                         # options={'maxiter': 4000, 'disp': True, 'gtol': 1e-15}).x

# store learned theta in thetahat
thetahat.theta = optresults.x

# print results and write to aggresultstxt
print('Optimization success:', optresults.success)
aggresultstxt.write(str(optresults.success) + delim)

print('Value of objective:', optresults.fun)
aggresultstxt.write(str(optresults.fun) + delim)

print('Number of iterations:', optresults.nit)
aggresultstxt.write(str(optresults.nit) + delim)

timeopt = time_ns() - timeoptstart
print('Time to optimize (ns):', timeopt)
aggresultstxt.write(str(timeopt) + delim)

# save the learned theta
np.save(workdir / f'thetahat-{cmdlineargmodel}-{nummodel}', thetahat.theta)
print('thetahat saved.')


###############################################################
# compute and graph results
###############################################################

# transform learned theta to real space potential
vlearnrec = thetahat.tox()

# learned potential vs initial potential
plt.plot(xvec, vlearnrec, '.-', label='Learned')
plt.plot(xvec, vinitrec, label='Initial')
plt.xlabel('x')
plt.title('Learned vs. Initial Potentials')
plt.legend()
# plt.show()
plt.savefig(resultsdir / f'graph_{scriptID}_{cmdlineargmodel}_{nummodel}_learned_vs_initial_potential.pdf', format='pdf')
plt.close()

# learned potential vs true potential
plt.plot(xvec, vlearnrec, '.-', label='Learned')
plt.plot(xvec, vtruexvec, label='True')
plt.xlabel('x')
plt.title('Learned vs. True Potentials')
plt.legend()
# plt.show()
plt.savefig(resultsdir / f'graph_{scriptID}_{cmdlineargmodel}_{nummodel}_true_vs_learned_potential.pdf', format='pdf')
plt.close()

print('')  # blank line

# shifted learned potential vs true potential
midpointindex = numx // 2
print('midpointindex =', midpointindex)
shift = vtruexvec[midpointindex] - vlearnrec[midpointindex]

# calculate and return l2 error
print('l2 error of learned potential:', nl.norm(vlearnrec - vtruexvec), sep='\n')
print('l2 error of shifted learned potential:', nl.norm(vlearnrec + shift - vtruexvec), sep='\n')
l2errshifttrim = nl.norm(vlearnrec[trim:-trim] + shift - vtruexvec[trim:-trim])
print('l2 error of shifted and trimmed learned potential:', l2errshifttrim, sep='\n')
aggresultstxt.write(str(l2errshifttrim) + delim)

print('')  # blank line

# calculate and return l2 error
print('l-inf error of learned potential:', np.amax(np.abs(vlearnrec - vtruexvec)), sep='\n')
print('l-inf error of shifted learned potential:', np.amax(np.abs(vlearnrec + shift - vtruexvec)), sep='\n')
linferrshifttrim = np.amax(np.abs(vlearnrec[trim:-trim] + shift - vtruexvec[trim:-trim]))
print('l-inf error of shifted and trimmed learned potential:', linferrshifttrim, sep='\n')
aggresultstxt.write(str(linferrshifttrim) + delim)

# plot shifted potential
plt.plot(xvec, vlearnrec + shift, '.-', label='Learned')
plt.plot(xvec, vtruexvec, label='True')
plt.xlabel('x')
plt.title(f'Shifted Learned Potential vs. True Potential\nl2 error (shift/trim) = {l2errshifttrim}\nl-inf error (shift/trim) = {linferrshifttrim}')
plt.legend()
# plt.show()
plt.savefig(resultsdir / f'graph_{scriptID}_{cmdlineargmodel}_{nummodel}_shifted_true_vs_learned_potential.pdf', format='pdf')
plt.close()

print('')  # blank line

# time to optimize
timetotal = time_ns()-timetotalstart
print('Total time of execution (ns):', timetotal)
# last item written to aggresultstxt so specify new line,
# this will insure that the next item added to the list is
# on its own line
aggresultstxt.write(str(timetotal) + newline)

# close aggresultstxt
aggresultstxt.close()

print('')  # blank line
