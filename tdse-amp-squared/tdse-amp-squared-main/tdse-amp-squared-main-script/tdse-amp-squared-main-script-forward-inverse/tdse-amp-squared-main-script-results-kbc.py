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

# get path to directory containing rsltadjthetarnd from command line
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

# fourtox = np.load(cwddir / 'fourtox.npy')
# vtoeptrue = np.load(cwddir / 'vtoeptrue.npy')
vxvec = np.load(cwddir / 'vxvec.npy')

# load learned theta
rsltadjthetarnd = np.load(cwddir / 'rsltadjthetarnd.npy')

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
# a0vec = amattruevec[:, 0]
# print('Shape a0vec:', a0vec.shape)

# make kinetic operator in the Fourier representation
# (this is constant for a given system)
kmat = np.diag(np.arange(-numfour, numfour + 1) ** 2 * np.pi ** 2 / (2 * L ** 2))


###############################################################
# transform learned theta (i.e., vhatmat) to real space potential
###############################################################

adjvtoeplearnR = rsltadjthetarnd[:numtoepelms]
adjvtoeplearnI = jnp.concatenate((jnp.array([0.0]), rsltadjthetarnd[numtoepelms:]))
adjvtoeplearn = adjvtoeplearnR + 1j * adjvtoeplearnI
adjvlearnfour = np.sqrt(2 * L) * np.concatenate([np.conjugate(np.flipud(adjvtoeplearn[1:(numfour + 1)])), adjvtoeplearn[:(numfour + 1)]])
adjvlearnrec = adjvlearnfour @ fourtox


###############################################################
# transform learned theta (i.e., vhatmat) to real space potential
###############################################################

# plot learned potential vs true potential
plt.plot(xvec, jnp.real(adjvlearnrec), '.-', label='adj')
plt.plot(xvec, vxvec, label='truth')
# plt.plot(xvec, jnp.real(vinitrec), label='init')
plt.xlabel('x')
plt.title('True Potential vs. Learned Potential')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_true_vs_learned_potential.pdf', format='pdf')
plt.close()

# plot shifted learned potential
zeroindex = len(xvec) // 2
adjdiff = np.abs(vxvec[zeroindex] - jnp.real(adjvlearnrec)[zeroindex])
plt.plot(xvec, jnp.real(adjvlearnrec) + adjdiff, '.-', label='adj')
plt.plot(xvec, vxvec, label='truth')
# plt.plot(xvec, jnp.real(vinitrec), label='init')
plt.xlabel('x')
plt.title('True Potential vs. Shifted Learned Potential')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_shifted_true_vs_learned_potential.pdf', format='pdf')
plt.close()

trim = np.where(xvec >= -10)  # 125
print('trim type:', type(trim))
print('len trim:', len(trim))
trim = trim[0][0]
print('trim =', trim)

print('l2 error of shifted adj potential:', nl.norm(jnp.real(adjvlearnrec) + adjdiff - vxvec), sep='\n')
print('l2 error of shifted and trimmed adj potential:', nl.norm(jnp.real(adjvlearnrec)[trim:-trim] + adjdiff - vxvec[trim:-trim]), sep='\n')
print('l-inf error of shifted and trimmed adj potential:', np.mean(np.abs(jnp.real(adjvlearnrec)[trim:-trim] + adjdiff - vxvec[trim:-trim])), sep='\n')