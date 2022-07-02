import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nl
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from jax.config import config
config.update("jax_enable_x64", True)

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'


###############################################################
# identify script on stdout
###############################################################

print('-------PROP TEST-------')


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
# amattruevec = np.load(cwddir / 'amattruevec.npy')

# fourtox = np.load(cwddir / 'fourtox.npy')
# vtoeptrue = np.load(cwddir / 'vtoeptrue.npy')
# vxvec = np.load(cwddir / 'vxvec.npy')

thetabestv = np.load(cwddir / 'thetabestv.npy')
thetabestprop = np.load(cwddir / 'thetabestprop.npy')

print('Computational environment loaded.')


###############################################################
# recreate variables from loaded data
###############################################################

# real space grid points (for plotting)
xvec = np.linspace(-L, L, numx)
trim = np.where(xvec >= -10)[0][0]  # 125
print('trim =', trim)

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

# compute propagator from thetabestv and thetabestprop

propbestv = thetatopropmat(thetabestv)
propbestprop = thetatopropmat(thetabestprop)


###############################################################
# Propagation test
#   Propagate a0vec with the true potential (propatrue) and
#   thetabest (theta which produced the lowest l2 error of
#   shifted and trimmed potential) past the training data
#   - propagate with the trimmed potential
#   - propagate with the trimmed and shifted potential
#       - l2 and l-inf errors of amat
#       - l2 and l-inf errors of psimat
#       - plot stepwise error l2 and l-inf errors of amat and psimat
###############################################################

# set multiplier of numts
tsmultiplier = 1
proptimesteps = np.arange(int(numts * tsmultiplier)) * dt
print('Final time for propagation:', proptimesteps[-1])

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
# results
###############################################################

print('l2 error of ahatmatvecbestv:', nl.norm(amattruevec - ahatmatvecbestv), sep='\n')
print('l-inf error of ahatmatvecbestv:', np.amax(np.abs(amattruevec - ahatmatvecbestv)), sep='\n')
print('l2 error of ahatmatvecbestprop:', nl.norm(amattruevec - ahatmatvecbestprop), sep='\n')
print('l-inf error of ahatmatvecbestprop:', np.amax(np.abs(amattruevec - ahatmatvecbestprop)), sep='\n')

l2errahatmatvecbestvstep = nl.norm(amattruevec - ahatmatvecbestv, axis=2)
l2errahatmatvecbestpropstep = nl.norm(amattruevec - ahatmatvecbestprop, axis=2)

print('Shape proptimesteps:', proptimesteps.shape)
print('Shape l2errahatmatvecbestvstep:', l2errahatmatvecbestvstep.shape)
print('Shape l2errahatmatvecbestpropstep:', l2errahatmatvecbestpropstep.shape)

for i in range(a0vec.shape[0]):
    plt.plot(proptimesteps, l2errahatmatvecbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, l2errahatmatvecbestpropstep[i], label=f'best propagation {i}')
    plt.title('Step-Wise l2 Error of Propagations - Fourier Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(cwddir / 'graph_step-wise_l2_error_amat_progation.pdf', format='pdf')
plt.close()

psimattruevec = amattruevec @ fourtox
psihatmatvecbestv = ahatmatvecbestv @ fourtox
psihatmatvecbestprop = ahatmatvecbestprop @ fourtox

print('l2 error of psihatmatvecbestv:', nl.norm(psimattruevec - psihatmatvecbestv), sep='\n')
print('l-inf error of psihatmatvecbestv:', np.amax(np.abs(psimattruevec - psihatmatvecbestv)), sep='\n')
print('l2 error of psihatmatvecbestprop:', nl.norm(psimattruevec - psihatmatvecbestprop), sep='\n')
print('l-inf error of psihatmatvecbestprop:', np.amax(np.abs(psimattruevec - psihatmatvecbestprop)), sep='\n')

l2errpsihatmatvecbestvstep = nl.norm(psimattruevec - psihatmatvecbestv, axis=2)
print('Shape l2errpsihatmatvecbestvstep:', l2errpsihatmatvecbestvstep.shape)
l2errpsihatmatvecbestpropstep = nl.norm(psimattruevec - psihatmatvecbestprop, axis=2)
print('Shape stepl2errpsihatmatvec:', l2errpsihatmatvecbestpropstep.shape)

for i in range(a0vec.shape[0]):
    plt.plot(proptimesteps, l2errpsihatmatvecbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, l2errpsihatmatvecbestpropstep[i], label=f'best propagation {i}')
    plt.title('Step-Wise l2 Error of Propagations - Real Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(cwddir / 'graph_step-wise_l2_error_psimat_progation.pdf', format='pdf')
plt.close()

print('Shape psimattruevec:', psimattruevec.shape)
print('Shape psihatmatvecbestv:', psihatmatvecbestv.shape)
print('Shape psihatmatvecbestprop:', psihatmatvecbestprop.shape)
print('Shape psimattruevec[:,:,trim:-trim]:', psimattruevec[:,:,trim:-trim].shape)
print('Shape psihatmatvecbestv[:,:,trim:-trim]:', psihatmatvecbestv[:,:,trim:-trim].shape)
print('Shape psihatmatvecbestprop[:,:,trim:-trim]:', psihatmatvecbestprop[:,:,trim:-trim].shape)

print('l2 error of trimmed psihatmatvecbestv:', nl.norm(psimattruevec[:,:,trim:-trim] - psihatmatvecbestv[:,:,trim:-trim]), sep='\n')
print('l-inf error of trimmed psihatmatvecbestv:', np.amax(np.abs(psimattruevec[:,:,trim:-trim] - psihatmatvecbestv[:,:,trim:-trim])), sep='\n')
print('l2 error of trimmed psihatmatvecbestprop:', nl.norm(psimattruevec[:,:,trim:-trim] - psihatmatvecbestprop[:,:,trim:-trim]), sep='\n')
print('l-inf error of trimmed psihatmatvecbestprop:', np.amax(np.abs(psimattruevec[:,:,trim:-trim] - psihatmatvecbestprop[:,:,trim:-trim])), sep='\n')

triml2errpsihatmatvecbestvstep = nl.norm(psimattruevec[:,:,trim:-trim] - psihatmatvecbestv[:,:,trim:-trim], axis=2)
print('Shape l2errpsihatmatvecbestvstep:', triml2errpsihatmatvecbestvstep.shape)
triml2errpsihatmatvecbestpropstep = nl.norm(psimattruevec[:,:,trim:-trim] - psihatmatvecbestprop[:,:,trim:-trim], axis=2)
print('Shape stepl2errpsihatmatvec:', triml2errpsihatmatvecbestpropstep.shape)

for i in range(a0vec.shape[0]):
    plt.plot(proptimesteps, triml2errpsihatmatvecbestvstep[i], label=f'best v {i}')
    plt.plot(proptimesteps, triml2errpsihatmatvecbestpropstep[i], label=f'best propagation {i}')
    plt.title('Step-Wise l2 Error of Propagations - Trimmed Real Space')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

plt.savefig(cwddir / 'graph_step-wise_l2_error_psimat_progation_trim.pdf', format='pdf')
plt.close()