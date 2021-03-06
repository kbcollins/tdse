import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import scipy.special as ss
import scipy.integrate as si
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from jax.config import config
config.update("jax_enable_x64", True)

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'


###############################################################
# identify script on stdout
###############################################################

print('-------PROP-------')
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
savename = 'propagate'

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
propatrue = np.load(workingdir / 'propatrue.npy')
# amattruevec = np.load(workdir / 'amattruevec.npy')

# load true potential
vtruetoep = np.load(workingdir / 'vtruetoep.npy')
# vtruexvec = np.load(workdir / 'vtruexvec.npy')

# load best learned potentials (i.e., \theta)
thetabestv = np.load(workingdir / 'thetabestv.npy')
thetabestprop = np.load(workingdir / 'thetabestprop.npy')

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

# construct initial state vector
# a0vec = amattruevec[:, 0]
# print('Shape a0vec:', a0vec.shape)

# make kinetic operator in the Fourier representation
# (this is constant for a given system)
kmat = np.diag(np.arange(-numfour, numfour + 1) ** 2 * np.pi ** 2 / (2 * L ** 2))

# make vtruemat
vtruemat = sl.toeplitz(r=vtruetoep, c=np.conj(vtruetoep))


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
# Functions for computing the propagator matrix given some theta
#   - theta is a vector containing the concatenation
#     of the real and imaginary parts of vmat
#   - theta should contain 2 * numtoepelms - 1
#     = 4 * numfour + 1 elements
###############################################################

def thetatovmat(theta):
    # to use theta we need to first recombine the real
    # and imaginary parts into a vector of complex values
    vtoephatR = theta[:numtoepelms]
    vtoephatI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
    vtoephat = vtoephatR + 1j * vtoephatI

    # construct vmathat from complex toeplitz vector
    vmathat = jnp.concatenate([jnp.flipud(jnp.conj(vtoephat)), vtoephat[1:]])[toepindxmat]

    return vmathat


def vmattopropmat(vmat):
    # Construct Hamiltonian matrix
    hmathat = kmat + vmat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hmathat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T

    return propahat


# compute vhatmat matrices for thetabestv and
# thetabestprop using thetatovmat
vhatmatbestv = thetatovmat(thetabestv)
vhatmatbestprop = thetatovmat(thetabestprop)

# compute propagator matrices for thetabestv and
# thetabestprop using thetatopropmat, vhatmatbestv
# and vhatmatbestprop
propbestv = vmattopropmat(vhatmatbestv)
propbestprop = vmattopropmat(vhatmatbestprop)


###############################################################
# propagate past training data
#   Propagate a0vec with the true potential (propatrue) and
#   thetabest (theta which produced the lowest l2 error of
#   shifted and trimmed potential) past the training data
#   - print l2 and l-inf errors of amat, plot stepwise l2 and
#     l-inf errors of amat
#   - print l2 and l-inf errors of psimat, plot stepwise l2 and
#     l-inf errors of psimat
#   - print l2 and l-inf errors of trimmed psimat, plot stepwise
#     l2 and l-inf errors of trimmed psimat
###############################################################

print('TRAINING SET RESULTS')
print('')  # blank line

# set multiplier of numts
tsmultiplier = 10  # 10
print('numts multiplier =', tsmultiplier)
proptimesteps = np.arange(int(numts * tsmultiplier)) * dt
print('Final time for propagation:', proptimesteps[-1])
print('')  # blank line

# propagate system starting from initial "a" state
# using the Hamiltonian constructed from the true potential
# (used for generating training data)
amattruevec = []
ahatmatvecbestv = []
ahatmatvecbestprop = []
for thisa0 in a0vec:
    tempamattrue = [thisa0.copy()]
    tempahatmatvecbestv = [thisa0.copy()]
    tempahatmatvecbestprop = [thisa0.copy()]
    for _ in range(proptimesteps.shape[0] - 1):
        tempamattrue.append(propatrue @ tempamattrue[-1])
        tempahatmatvecbestv.append(propbestv @ tempahatmatvecbestv[-1])
        tempahatmatvecbestprop.append(propbestprop @ tempahatmatvecbestprop[-1])

    amattruevec.append(tempamattrue)
    ahatmatvecbestv.append(tempahatmatvecbestv)
    ahatmatvecbestprop.append(tempahatmatvecbestprop)


amattruevec = jnp.array(amattruevec)
ahatmatvecbestv = jnp.array(ahatmatvecbestv)
ahatmatvecbestprop = jnp.array(ahatmatvecbestprop)


###############################################################
# results of propagating past training data
###############################################################

# l2 and l-inf errors for amat
print('l2 error of ahatmatvecbestv:', nl.norm(amattruevec - ahatmatvecbestv), sep='\n')
print('l-inf error of ahatmatvecbestv:', np.amax(np.abs(amattruevec - ahatmatvecbestv)), sep='\n')
print('l2 error of ahatmatvecbestprop:', nl.norm(amattruevec - ahatmatvecbestprop), sep='\n')
print('l-inf error of ahatmatvecbestprop:', np.amax(np.abs(amattruevec - ahatmatvecbestprop)), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for amat
l2errahatmatvecbestvstep = nl.norm(amattruevec - ahatmatvecbestv, axis=2)
l2errahatmatvecbestpropstep = nl.norm(amattruevec - ahatmatvecbestprop, axis=2)
for i in range(a0vec.shape[0]):
    plt.plot(proptimesteps, l2errahatmatvecbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, l2errahatmatvecbestpropstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Fourier Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_step-wise_l2_error_amat_progation.pdf', format='pdf')
plt.close()

# transform amat to real space wave functions
psimattruevec = amattruevec @ fourtox
psihatmatvecbestv = ahatmatvecbestv @ fourtox
psihatmatvecbestprop = ahatmatvecbestprop @ fourtox

# l2 and l-inf errors for psimat
print('l2 error of psihatmatvecbestv:', nl.norm(psimattruevec - psihatmatvecbestv), sep='\n')
print('l-inf error of psihatmatvecbestv:', np.amax(np.abs(psimattruevec - psihatmatvecbestv)), sep='\n')
print('l2 error of psihatmatvecbestprop:', nl.norm(psimattruevec - psihatmatvecbestprop), sep='\n')
print('l-inf error of psihatmatvecbestprop:', np.amax(np.abs(psimattruevec - psihatmatvecbestprop)), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for psimat
l2errpsihatmatvecbestvstep = nl.norm(psimattruevec - psihatmatvecbestv, axis=2)
l2errpsihatmatvecbestpropstep = nl.norm(psimattruevec - psihatmatvecbestprop, axis=2)
for i in range(a0vec.shape[0]):
    plt.plot(proptimesteps, l2errpsihatmatvecbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, l2errpsihatmatvecbestpropstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Real Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_step-wise_l2_error_psimat_progation.pdf', format='pdf')
plt.close()

# l2 and l-inf errors for trimmed psimat
print('l2 error of trimmed psihatmatvecbestv:', nl.norm(psimattruevec[:,:,trim:-trim] - psihatmatvecbestv[:,:,trim:-trim]), sep='\n')
print('l-inf error of trimmed psihatmatvecbestv:', np.amax(np.abs(psimattruevec[:,:,trim:-trim] - psihatmatvecbestv[:,:,trim:-trim])), sep='\n')
print('l2 error of trimmed psihatmatvecbestprop:', nl.norm(psimattruevec[:,:,trim:-trim] - psihatmatvecbestprop[:,:,trim:-trim]), sep='\n')
print('l-inf error of trimmed psihatmatvecbestprop:', np.amax(np.abs(psimattruevec[:,:,trim:-trim] - psihatmatvecbestprop[:,:,trim:-trim])), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for trimmed psimat
triml2errpsihatmatvecbestvstep = nl.norm(psimattruevec[:,:,trim:-trim] - psihatmatvecbestv[:,:,trim:-trim], axis=2)
triml2errpsihatmatvecbestpropstep = nl.norm(psimattruevec[:,:,trim:-trim] - psihatmatvecbestprop[:,:,trim:-trim], axis=2)
for i in range(a0vec.shape[0]):
    plt.plot(proptimesteps, triml2errpsihatmatvecbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, triml2errpsihatmatvecbestpropstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Trimmed Real Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_step-wise_l2_error_psimat_progation_trim.pdf', format='pdf')
plt.close()

print('')  # blank line


###############################################################
# propagate a0 not used for training
#   Start with 3 wave functions, not explicitly used for
#   training. Generate a0testset and propagate with thetatrue,
#   thetabestv, and thetabestprop
#   shifted and trimmed potential) past the training data
#   - print l2 and l-inf errors of amat, plot stepwise l2 and
#     l-inf errors of amat
#   - print l2 and l-inf errors of psimat, plot stepwise l2 and
#     l-inf errors of psimat
#   - print l2 and l-inf errors of trimmed psimat, plot stepwise
#     l2 and l-inf errors of trimmed psimat
###############################################################

print('TEST SET RESULTS')
print('')  # blank line

# define initial state functions
def psicheb(x):
    return ss.eval_chebyt(10, x/L)

def psicmplx(x):
    yexpvec1 = np.exp(-(x + 4) ** 2 / 20)
    yexpvec2 = -np.exp(-(x - 5) ** 2 / 20)
    return yexpvec1 + 1j*yexpvec2

def psisquare(x):
    # THIS CAUSES THE QUAD FUNCTION TO RETURN AN ERROR
    # IT NEEDS TO BE FIXED
    if len(x) == 1:
        if x >= -7 and x <=-3:
            return 1
        else:
            return 0
    else:
        return np.logical_and(x >= -7, x <=-3) * 1


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


# initial state vector
psi0testset = [psicheb, psicmplx]  # [psicheb, psicmplx, psisquare]

# make initial states, a0, with mka0
a0testset = []
for thispsi0fn in psi0testset:
    thisa0, _ = mka0(thispsi0fn)
    a0testset.append(thisa0)

print('Number of a0testset states:', len(a0testset))

# propagate
amattruetestset = []
ahatmattestsetbestv = []
ahatmattestsetbestprop = []
for thisa0 in a0testset:
    tempamattrue = [thisa0.copy()]
    tempahatmatvecbestv = [thisa0.copy()]
    tempahatmatvecbestprop = [thisa0.copy()]
    for _ in range(proptimesteps.shape[0] - 1):
        tempamattrue.append(propatrue @ tempamattrue[-1])
        tempahatmatvecbestv.append(propbestv @ tempahatmatvecbestv[-1])
        tempahatmatvecbestprop.append(propbestprop @ tempahatmatvecbestprop[-1])

    amattruetestset.append(tempamattrue)
    ahatmattestsetbestv.append(tempahatmatvecbestv)
    ahatmattestsetbestprop.append(tempahatmatvecbestprop)


amattruetestset = jnp.array(amattruetestset)
ahatmattestsetbestv = jnp.array(ahatmattestsetbestv)
ahatmattestsetbestprop = jnp.array(ahatmattestsetbestprop)


###############################################################
# results of propagating out-of-training wave functions
###############################################################

# l2 and l-inf errors for amat
print('l2 error of ahatmattestsetbestv:', nl.norm(amattruetestset - ahatmattestsetbestv), sep='\n')
print('l-inf error of ahatmattestsetbestv:', np.amax(np.abs(amattruetestset - ahatmattestsetbestv)), sep='\n')
print('l2 error of ahatmattestsetbestprop:', nl.norm(amattruetestset - ahatmattestsetbestprop), sep='\n')
print('l-inf error of ahatmattestsetbestprop:', np.amax(np.abs(amattruetestset - ahatmattestsetbestprop)), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for amat
l2errahatmattestsetbestvstep = nl.norm(amattruetestset - ahatmattestsetbestv, axis=2)
l2errahatmattestsetbestpropstep = nl.norm(amattruetestset - ahatmattestsetbestprop, axis=2)
for i in range(len(a0testset)):
    plt.plot(proptimesteps, l2errahatmattestsetbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, l2errahatmattestsetbestpropstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Fourier Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_test-set_step-wise_l2_error_amat_progation.pdf', format='pdf')
plt.close()

# transform amat to real space wave functions
psimattruetestset = amattruetestset @ fourtox
psihatmattestsetbestv = ahatmattestsetbestv @ fourtox
psihatmattestsetbestprop = ahatmattestsetbestprop @ fourtox

# l2 and l-inf errors for psimat
print('l2 error of psihatmattestsetbestv:', nl.norm(psimattruetestset - psihatmattestsetbestv), sep='\n')
print('l-inf error of psihatmattestsetbestv:', np.amax(np.abs(psimattruetestset - psihatmattestsetbestv)), sep='\n')
print('l2 error of psihatmattestsetbestprop:', nl.norm(psimattruetestset - psihatmattestsetbestprop), sep='\n')
print('l-inf error of psihatmattestsetbestprop:', np.amax(np.abs(psimattruetestset - psihatmattestsetbestprop)), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for psimat
l2errpsihatmattestsetbestvstep = nl.norm(psimattruetestset - psihatmattestsetbestv, axis=2)
l2errpsihatmattestsetbestpropstep = nl.norm(psimattruetestset - psihatmattestsetbestprop, axis=2)
for i in range(len(a0testset)):
    plt.plot(proptimesteps, l2errpsihatmattestsetbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, l2errpsihatmattestsetbestpropstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Real Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_test-set_step-wise_l2_error_psimat_progation.pdf', format='pdf')
plt.close()

# l2 and l-inf errors for trimmed psimat
print('l2 error of trimmed psihatmattestsetbestv:', nl.norm(psimattruetestset[:,:,trim:-trim] - psihatmattestsetbestv[:,:,trim:-trim]), sep='\n')
print('l-inf error of trimmed psihatmattestsetbestv:', np.amax(np.abs(psimattruetestset[:,:,trim:-trim] - psihatmattestsetbestv[:,:,trim:-trim])), sep='\n')
print('l2 error of trimmed psihatmattestsetbestprop:', nl.norm(psimattruetestset[:,:,trim:-trim] - psihatmattestsetbestprop[:,:,trim:-trim]), sep='\n')
print('l-inf error of trimmed psihatmattestsetbestprop:', np.amax(np.abs(psimattruetestset[:,:,trim:-trim] - psihatmattestsetbestprop[:,:,trim:-trim])), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for trimmed psimat
triml2errpsihatmattestsetbestvstep = nl.norm(psimattruetestset[:,:,trim:-trim] - psihatmattestsetbestv[:,:,trim:-trim], axis=2)
triml2errpsihatmattestsetbestpropstep = nl.norm(psimattruetestset[:,:,trim:-trim] - psihatmattestsetbestprop[:,:,trim:-trim], axis=2)
for i in range(len(a0testset)):
    plt.plot(proptimesteps, triml2errpsihatmattestsetbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, triml2errpsihatmattestsetbestpropstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Trimmed Real Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_test-set_step-wise_l2_error_psimat_progation_trim.pdf', format='pdf')
plt.close()


###############################################################
# propagate using learned static potential + time-dependent
# perturbation
#   Start with psitestset and propagate with thetatrue,
#   thetabestv, and thetabestprop plus a small time-dependent
#   perturbation
#   - print l2 and l-inf errors of amat, plot stepwise l2 and
#     l-inf errors of amat
#   - print l2 and l-inf errors of psimat, plot stepwise l2 and
#     l-inf errors of psimat
#   - print l2 and l-inf errors of trimmed psimat, plot stepwise
#     l2 and l-inf errors of trimmed psimat
###############################################################

def vtd(t):
    return lambda z: -(t / proptimesteps[-1]) * np.exp(-(z-5)**2 / 8)

# propagate
amattruevtd = []
ahatmatbestvtd = []
ahatmatbestproptd = []
for thisa0 in a0testset:
    tempamattruevtd = [thisa0.copy()]
    tempahatmatbestvtd = [thisa0.copy()]
    tempahatmatbestproptd = [thisa0.copy()]

    for thist in range(proptimesteps[1:]):
        # compute the potential operator matrix, vmat, for this t
        thisvtdtoep = []
        thisvtd = vtd(thist)
        for thisfourn in range(numtoepelms):
            def intgrnd(x):
                return thisvtd(x) * np.exp(-1j * np.pi * thisfourn * x / L) / (2 * L)
            def rintgrnd(x):
                return intgrnd(x).real
            def iintgrnd(x):
                return intgrnd(x).imag

            thisvtdtoep.append(si.quad(rintgrnd, -L, L, limit=100)[0] + 1j * si.quad(iintgrnd, -L, L, limit=100)[0])

        thisvtdmat = sl.toeplitz(r=thisvtdtoep, c=np.conj(thisvtdtoep))

        # construct propagation matrices
        propatruetd = vmattopropmat(vtruemat + thisvtdmat)
        propbestvtd = vmattopropmat(vhatmatbestv + thisvtdmat)
        propbestproptd = vmattopropmat(vhatmatbestprop + thisvtdmat)

        tempamattruevtd.append(propatruetd @ tempamattruevtd[-1])
        tempahatmatbestvtd.append(propbestvtd @ tempahatmatbestvtd[-1])
        tempahatmatbestproptd.append(propbestproptd @ tempahatmatbestproptd[-1])

    amattruevtd.append(tempamattruevtd)
    ahatmatbestvtd.append(tempahatmatbestvtd)
    ahatmatbestproptd.append(tempahatmatbestproptd)

amattruevtd = jnp.array(amattruevtd)
ahatmatbestvtd = jnp.array(ahatmatbestvtd)
ahatmatbestproptd = jnp.array(ahatmatbestproptd)


###############################################################
# results of propagating with time-dependent perturbation
###############################################################

# l2 and l-inf errors for amat
print('l2 error of ahatmatbestvtd:', nl.norm(amattruevtd - ahatmatbestvtd), sep='\n')
print('l-inf error of ahatmatbestvtd:', np.amax(np.abs(amattruevtd - ahatmatbestvtd)), sep='\n')
print('l2 error of ahatmatbestproptd:', nl.norm(amattruevtd - ahatmatbestproptd), sep='\n')
print('l-inf error of ahatmatbestproptd:', np.amax(np.abs(amattruevtd - ahatmatbestproptd)), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for amat
l2errahatmatbestvtdstep = nl.norm(amattruevtd - ahatmatbestvtd, axis=2)
l2errahatmatbestproptdstep = nl.norm(amattruevtd - ahatmatbestproptd, axis=2)
for i in range(len(a0testset)):
    plt.plot(proptimesteps, l2errahatmatbestvtdstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, l2errahatmatbestproptdstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Fourier Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_vtd-pert_step-wise_l2_error_amat_progation.pdf', format='pdf')
plt.close()

# transform amat to real space wave functions
psimattruevtd = amattruevtd @ fourtox
psihatmatbestvtd = ahatmatbestvtd @ fourtox
psihatmatbestproptd = ahatmatbestproptd @ fourtox

# l2 and l-inf errors for psimat
print('l2 error of psihatmatbestvtd:', nl.norm(psimattruevtd - psihatmatbestvtd), sep='\n')
print('l-inf error of psihatmatbestvtd:', np.amax(np.abs(psimattruevtd - psihatmatbestvtd)), sep='\n')
print('l2 error of psihatmatbestproptd:', nl.norm(psimattruevtd - psihatmatbestproptd), sep='\n')
print('l-inf error of psihatmatbestproptd:', np.amax(np.abs(psimattruevtd - psihatmatbestproptd)), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for psimat
l2errpsihatmatbestvtdstep = nl.norm(psimattruevtd - psihatmatbestvtd, axis=2)
l2errpsihatmatbestproptdstep = nl.norm(psimattruevtd - psihatmatbestproptd, axis=2)
for i in range(len(a0testset)):
    plt.plot(proptimesteps, l2errpsihatmatbestvtdstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, l2errpsihatmatbestproptdstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Real Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_vtd-pert_step-wise_l2_error_psimat_progation.pdf', format='pdf')
plt.close()

# l2 and l-inf errors for trimmed psimat
print('l2 error of trimmed psihatmatbestvtd:', nl.norm(psimattruevtd[:,:,trim:-trim] - psihatmatbestvtd[:,:,trim:-trim]), sep='\n')
print('l-inf error of trimmed psihatmatbestvtd:', np.amax(np.abs(psimattruevtd[:,:,trim:-trim] - psihatmatbestvtd[:,:,trim:-trim])), sep='\n')
print('l2 error of trimmed psihatmatbestproptd:', nl.norm(psimattruevtd[:,:,trim:-trim] - psihatmatbestproptd[:,:,trim:-trim]), sep='\n')
print('l-inf error of trimmed psihatmatbestproptd:', np.amax(np.abs(psimattruevtd[:,:,trim:-trim] - psihatmatbestproptd[:,:,trim:-trim])), sep='\n')
print('')  # blank line

# plot of stepwise l2 and l-inf errors for trimmed psimat
triml2errpsihatmatbestvtdstep = nl.norm(psimattruevtd[:,:,trim:-trim] - psihatmatbestvtd[:,:,trim:-trim], axis=2)
triml2errpsihatmatbestproptdstep = nl.norm(psimattruevtd[:,:,trim:-trim] - psihatmatbestproptd[:,:,trim:-trim], axis=2)
for i in range(len(a0testset)):
    plt.plot(proptimesteps, triml2errpsihatmatbestvtdstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, triml2errpsihatmatbestproptdstep[i], label=f'best prop {i}')
    plt.title('Step-Wise l2 Error of Propagations - Trimmed Real Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(resultsdir / f'graph_{savename}_vtd-pert_step-wise_l2_error_psimat_progation_trim.pdf', format='pdf')
plt.close()

