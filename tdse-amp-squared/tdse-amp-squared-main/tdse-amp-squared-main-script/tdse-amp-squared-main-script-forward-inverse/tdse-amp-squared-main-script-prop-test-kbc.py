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
tsmultiplier = 5

# propagate system starting from initial "a" state
# using the Hamiltonian constructed from the true potential
# (used for generating training data)
amattruevec = []
amatlearnedvec = []
for thisa0 in a0vec:
    tempamattrue = [thisa0.copy()]
    tempamatlearned = [thisa0.copy()]
    for i in range(numts * tsmultiplier):
        tempamattrue.append(propatrue @ tempamattrue[-1])
        tempamatlearned.append(propatrue @ tempamatlearned[-1])

    amattruevec.append(tempamattrue)
    amatlearnedvec.append(tempamatlearned)

amattruevec = jnp.array(amattruevec)
psimattruevec = amattruevec @ fourtox
amatlearnedvec = jnp.array(amatlearnedvec)
psimatlearnedvec = amatlearnedvec @ fourtox

print('Done with propagation.')


###############################################################
# results
###############################################################

numitrs = 200
midpointindex = numx // 2
print('midpointindex =', midpointindex)
trim = np.where(xvec >= -10)[0][0]  # 125
print('trim =', trim)


np.save(cwddir / 'thetabest', thetabest)
print('thetabest saved.')

print('Mean rawl2err:', np.mean(rawl2err))
print('Minumum of rawl2err:', np.amin(rawl2err))
print('Maximum of rawl2err:', np.amax(rawl2err))
print('Average deviation of rawl2err:', np.mean(np.abs(np.subtract(rawl2err, np.mean(rawl2err)))))

print('Mean rawlinferr:', np.mean(rawlinferr))
print('Minumum of rawlinferr:', np.amin(rawlinferr))
print('Maximum of rawlinferr:', np.amax(rawlinferr))
print('Average deviation of rawlinferr:', np.mean(np.abs(np.subtract(rawlinferr, np.mean(rawlinferr)))))

print('Mean shiftl2err:', np.mean(shiftl2err))
print('Minumum of shiftl2err:', np.amin(shiftl2err))
print('Maximum of shiftl2err:', np.amax(shiftl2err))
print('Average deviation of shiftl2err:', np.mean(np.abs(np.subtract(shiftl2err, np.mean(shiftl2err)))))

print('Mean shiftlinferr:', np.mean(shiftlinferr))
print('Minumum of shiftlinferr:', np.amin(shiftlinferr))
print('Maximum of shiftlinferr:', np.amax(shiftlinferr))
print('Average deviation of shiftlinferr:', np.mean(np.abs(np.subtract(shiftlinferr, np.mean(shiftlinferr)))))

print('Mean trimshiftl2err:', np.mean(trimshiftl2err))
print('Minumum of trimshiftl2err:', np.amin(trimshiftl2err))
print('Maximum of trimshiftl2err:', np.amax(trimshiftl2err))
print('Average deviation of trimshiftl2err:', np.mean(np.abs(np.subtract(trimshiftl2err, np.mean(trimshiftl2err)))))

print('Mean trimshiftlinferr:', np.mean(trimshiftlinferr))
print('Minumum of trimshiftlinferr:', np.amin(trimshiftlinferr))
print('Maximum of trimshiftlinferr:', np.amax(trimshiftlinferr))
print('Average deviation of trimshiftlinferr:', np.mean(np.abs(np.subtract(trimshiftlinferr, np.mean(trimshiftlinferr)))))

plt.plot(rawl2err, label='rawl2err')
plt.plot(shiftl2err, label='shiftl2err')
plt.plot(trimshiftl2err, label='trimshiftl2err')
plt.title('l2 Error')
plt.xlabel('Trial Number')
plt.legend()
plt.savefig(cwddir / 'graph_l2_error.pdf', format='pdf')
plt.close()

plt.plot(rawl2err, label='rawlinferr')
plt.plot(shiftl2err, label='shiftlinferr')
plt.plot(trimshiftl2err, label='trimshiftlinferr')
plt.title('l-infinite Error')
plt.xlabel('Trial Number')
plt.legend()
plt.savefig(cwddir / 'graph_l-infinite_error.pdf', format='pdf')
plt.close()