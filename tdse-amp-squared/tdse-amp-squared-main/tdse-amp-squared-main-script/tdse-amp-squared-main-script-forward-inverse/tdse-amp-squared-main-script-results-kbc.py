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
amattruevec = np.load(cwddir / 'amattruevec.npy')

vxvec = np.load(cwddir / 'vxvec.npy')

# load initial theta
thetarnd = np.load(cwddir / 'thetarnd.npy')

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
# used like realspacevec = fourspacevec @ fourtox
fourtox = np.exp(1j * np.pi * np.outer(fournvec, xvec) / L) / np.sqrt(2 * L)

# number of Toeplitz elements in the Fourier representation
numtoepelms = 2 * numfour + 1

# make kinetic operator in the Fourier representation
# (this is constant for a given system)
kmat = np.diag(np.arange(-numfour, numfour + 1) ** 2 * np.pi ** 2 / (2 * L ** 2))


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

# transform init theta (i.e., initvhatmat) to real space potential
vinitrec = thetatoreal(thetarnd)

# transform learned theta to real space potential
vlearnrec = thetatoreal(rsltadjthetarnd)


###############################################################
# transform learned theta (i.e., vhatmat) to real space potential
###############################################################

# learned potential vs initial potential
plt.plot(xvec, jnp.real(vlearnrec), '.-', label='Learned')
plt.plot(xvec, jnp.real(vinitrec), label='Initial')
plt.xlabel('x')
plt.title('Learned vs. Initial Potentials')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_learned_vs_initial_potential.pdf', format='pdf')
plt.close()

# learned potential vs true potential
print('l2 error of learned potential:', nl.norm(jnp.real(vlearnrec) - vxvec), sep='\n')
print('l-inf error of learned potential:', np.mean(np.abs(jnp.real(vlearnrec) - vxvec)), sep='\n')
plt.plot(xvec, jnp.real(vlearnrec), '.-', label='Learned')
plt.plot(xvec, vxvec, label='True')
plt.xlabel('x')
plt.title('Learned vs. True Potentials')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_true_vs_learned_potential.pdf', format='pdf')
plt.close()

# shifted learned potential vs true potential
# zeroindex = np.where(xvec == 0)[0][0]
# zeroindex = len(xvec) // 2
midpointindex = numx // 2
print('midpointindex =', midpointindex)
# shift = vxvec[zeroindex] - jnp.real(vlearnrec)[zeroindex]
shift = vxvec[midpointindex] - jnp.real(vlearnrec)[midpointindex]
print('l2 error of shifted learned potential:', nl.norm(jnp.real(vlearnrec) + shift - vxvec), sep='\n')
print('l-inf error of shifted learned potential:', np.mean(np.abs(jnp.real(vlearnrec) + shift - vxvec)), sep='\n')
plt.plot(xvec, jnp.real(vlearnrec) + shift, '.-', label='Learned')
plt.plot(xvec, vxvec, label='True')
plt.xlabel('x')
plt.title('Shifted Learned Potential vs. True Potential')
plt.legend()
# plt.show()
plt.savefig(cwddir / 'graph_shifted_true_vs_learned_potential.pdf', format='pdf')
plt.close()

# Shifted and trimmed learned potential vs true potential
trim = np.where(xvec >= -10)[0][0]  # 125
print('trim =', trim)
print('l2 error of shifted and trimmed learned potential:', nl.norm(jnp.real(vlearnrec)[trim:-trim] + shift - vxvec[trim:-trim]), sep='\n')
print('l-inf error of shifted and trimmed learned potential:', np.mean(np.abs(jnp.real(vlearnrec)[trim:-trim] + shift - vxvec[trim:-trim])), sep='\n')
