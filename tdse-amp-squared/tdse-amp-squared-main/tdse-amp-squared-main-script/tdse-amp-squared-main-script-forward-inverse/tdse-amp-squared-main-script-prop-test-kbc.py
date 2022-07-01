import sys
import pathlib
import numpy as np
import numpy.linalg as nl
import jax.numpy as jnp
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
tsmultiplier = 0.5

# propagate system starting from initial "a" state
# using the Hamiltonian constructed from the true potential
# (used for generating training data)
amattruevec = []
ahatmatvec = []
for thisa0 in a0vec:
    tempamattrue = [thisa0.copy()]
    tempamatlearned = [thisa0.copy()]
    for i in range(int(numts * tsmultiplier)):
        tempamattrue.append(propatrue @ tempamattrue[-1])
        tempamatlearned.append(propatrue @ tempamatlearned[-1])

    amattruevec.append(tempamattrue)
    ahatmatvec.append(tempamatlearned)


###############################################################
# results
###############################################################

amattruevec = jnp.array(amattruevec)
ahatmatvec = jnp.array(ahatmatvec)

print('l2 error of ahatmatvec:', nl.norm(amattruevec - ahatmatvec), sep='\n')
print('l-inf error of ahatmatvec:', np.amax(np.abs(amattruevec - ahatmatvec)), sep='\n')

stepl2errahatmatvec = nl.norm(amattruevec - ahatmatvec, axis=2)
print('Shape stepl2errahatmatvec:', stepl2errahatmatvec.shape)

# plt.title('l-infinite Error')
# plt.xlabel('Trial Number')
# plt.legend()
# plt.savefig(cwddir / 'graph_l-infinite_error.pdf', format='pdf')
# plt.close()

psimattruevec = amattruevec @ fourtox
psihatmatvec = ahatmatvec @ fourtox

print('l2 error of psihatmatvec:', nl.norm(psimattruevec - psihatmatvec), sep='\n')
print('l-inf error of ahatmatvec:', np.amax(np.abs(psimattruevec - psihatmatvec)), sep='\n')

stepl2errpsihatmatvec = nl.norm(psimattruevec - psihatmatvec, axis=2)
print('Shape stepl2errahatmatvec:', stepl2errpsihatmatvec.shape)
