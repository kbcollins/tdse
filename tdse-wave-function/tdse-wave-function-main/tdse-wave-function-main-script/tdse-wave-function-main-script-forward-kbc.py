import sys
import pathlib
import numpy as np
import scipy.linalg as sl
import scipy.integrate as si
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from jax.config import config
config.update("jax_enable_x64", True)

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import tdsemodelclass


###############################################################
# identify script on stdout
###############################################################

print('-------FORWARD-------')
print('')  # blank line


###############################################################
# get commandline arguments
# - cmdlineargsavedir: directory to save files to
# - cmdlineargpotential: selection of true potential,
#   possible selections are {0, 1, 2, 3, 4, 5, 6}
# - cmdlineargnumts: number of times steps of training
#   trajectories
# - cmdlineargdt: time-step size
###############################################################

print('sys.argv =', sys.argv)

# directory to load from/save to
cmdlineargsavedir = pathlib.Path(sys.argv[1])
print('cmdlineargsavedir =', cmdlineargsavedir)

# selection of true potential function
cmdlineargpotential = int(sys.argv[2])
print('cmdlineargpotential =', cmdlineargpotential)

# number of time steps
cmdlineargnumts = int(sys.argv[3])
print('cmdlineargnumts =', cmdlineargnumts)

# time-step size
cmdlineargdt = float(sys.argv[4])
print('cmdlineargdt =', cmdlineargdt)

# file path to output directory
workingdir = cmdlineargsavedir / f'v{cmdlineargpotential}'
print('Current working directory:', workingdir)
print('')  # blank line


###############################################################
# set computational parameters
###############################################################

# size of spatial domain
L = 15.0

# number of real space grid points (for plotting)
numx = 1025

# number of Fourier basis functions
numfour = 32  # 64

# set number of time steps
# trajectory's length = numts + 1
numts = cmdlineargnumts
# numts = 20  # 20

# set time-step size
dt = cmdlineargdt
# dt = 1e-2  # 1e-2

# print computational environment variables to stdout
print('L =', L)
print('numx =', numx)
print('numfour =', numfour)
print('numts =', numts)
print('dt =', dt)

# save computational parameters to disk
cmpprm = [L, numx, numfour, dt, numts]  # original cmpprm (what all other scripts expect)
np.save(workingdir / 'cmpprm', cmpprm)
print('Computational parameters saved.')

###############################################################
# set what model to use to approximate the potential and
# specify the parameters which fully define the model
###############################################################

# Fourier model
model = tdsemodelclass.fourier
print('model = fourier')
modelprms = (L, numx, numfour)

# Chebyshev model
# - From experience, I have found that the cheby model
#   works best when numcheb is an odd numbers
# model = tdsemodelclass.cheby
# print('model = cheby')
# numcheb = 11
# modelprms = (L, numx, numfour, numcheb)

print('')  # blank line


###############################################################
# utilities - created from the computational parameters
###############################################################

# real space grid points (for plotting)
xvec = np.linspace(-L, L, numx)

# vector of Fourier mode indices
# fournvec = -numfour,...,0,...,numfour
fournvec = np.arange(-numfour, numfour + 1)

# matrix for converting Fourier representation to real space
# used like realspacevec = fourspacevec @ fourtox
fourtox = np.exp(1j * np.pi * np.outer(fournvec, xvec) / L) / np.sqrt(2 * L)
# np.save(workingdir / 'fourtox', fourtox)
# print('fourtox saved.')

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
# true potential - used for generating training data
###############################################################

# define true potential (for generating training data)
if cmdlineargpotential == 0:
    def v(z):
        # harmonic oscillator potential (should be exact for Chebyshev)
        return 0.5 * z**2
elif cmdlineargpotential == 1:
    def v(z):
        # symmetric double well potential
        return 2.5e-3 * (z ** 2 - 25) ** 2
elif cmdlineargpotential == 2:
    def v(z):
        # asymmetric double well potential
        c0 = 4.35
        c1 = 9.40e-1
        c2 = -3.56e-1
        c3 = -4.66e-2
        c4 = 1.46e-2
        c5 = 6.76e-4
        c6 = -1.26e-4
        c7 = -5.43e-6
        c8 = 4.12e-7
        c9 = 1.65e-8
        x = z + 0.8
        return 0.5 * (c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4 + c5 * x ** 5 + c6 * x ** 6 + c7 * x ** 7 + c8 * x ** 8 + c9 * x ** 9)
elif cmdlineargpotential == 3:
    def v(z):
        # non-polynomial potentials
        return np.sin(0.4 * z - 1)
elif cmdlineargpotential == 4:
    def v(z):
        # non-polynomial potentials
        return np.sin((0.5 * z) ** 2)
elif cmdlineargpotential == 5:
    def v(z):
        # non-polynomial potentials
        return 15 * (-np.cos(z) + np.sin((0.5 * z) ** 2 - 0.2 * z))
elif cmdlineargpotential == 6:
    def v(z):
        # soft coulomb potential
        return -1 / np.sqrt(z ** 2 + 0.25)
else:
    print(f'Selection of "{cmdlineargpotential}" is not recognized as an available potential.')

# true potential on real space grid (for plotting)
vtruexvec = v(xvec)
np.save(workingdir / 'vtruexvec', vtruexvec)
print('vtruexvec saved.')


###############################################################
# model of true potential
###############################################################

# create a model object and save as thetatrue
# - modelprms is a tuple containing all of the variables the
#   model needs to be fully defined
# - the '*' in *modelprms unpacks modelprms then passes
#   everything as the parameters to the instantiation of a
#   model object
thetatrue = model(*modelprms)

# load thetatrue with the true potential represented in
# terms of the model
thetatrue.theta = model.fntotheta(v, *modelprms)
print('Shape thetatrue:', thetatrue.theta.shape)


###############################################################
# initial states - a0
###############################################################

# define initial state functions
def psi0_0(x):
    return 10 * np.exp(-((x + 3) / 4)**2) * (2.0 / np.pi)**0.25
    # return 10 * np.exp(-((x + 3) / 2)**2) * (2.0 / np.pi)**0.25

def psi0_1(x):
    return np.exp(-((x - 3) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-((x - 3) / 2)**2) * (2.0 / np.pi)**0.25

def psi0_2(x):
    # return np.exp(-x**2) * (2.0 / np.pi)**0.25
    return np.exp(-((x - 8) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-((x - 6)/4)**2) * (2.0 / np.pi)**0.25

def psi0_3(x):
    # a weird non-symmetric wavefunction
    # return np.abs(np.sin((0.15*x - 0.5)**2))
    return np.exp(-((x + 8) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-((x + 6)/4)**2) * (2.0 / np.pi)**0.25

def psi0_4(x):
    return np.exp(-((x - 12) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-(x - 11)**2) * (2.0 / np.pi)**0.25

def psi0_5(x):
    return np.exp(-((x + 12) / 4)**2) * (2.0 / np.pi)**0.25
    # return np.exp(-(x + 11)**2) * (2.0 / np.pi)**0.25


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


# generate initial state vector
# pick initial un-normalized wave functions
psi0fnvec = [psi0_0, psi0_1, psi0_2, psi0_3]  # [psi0_0, psi0_1, psi0_2, psi0_3, psi0_4, psi0_5]

# make initial states, a0, with mka0
a0vec = []
normpsi0xvec = []
normpsi0recxvec = []
for thispsi0fn in psi0fnvec:
    tempa0, tempnormpsi0x = mka0(thispsi0fn)
    a0vec.append(tempa0)
    # normpsi0xvec.append(tempnormpsi0x)
    # normpsi0recxvec.append(tempa0 @ fourtox)


print('Number of a0 states:', len(a0vec))

np.save(workingdir / 'a0vec', a0vec)
print('a0vec saved.')


###############################################################
# forward propagation - make training data
###############################################################

# **************************************************
# the following code enclosed by ' # ****' is the
# same regardless of the model use
# **************************************************
# Hamiltonian operator with true potential
# in the Fourier representation
hmattrue = kmat + thetatrue.tovmat()

# eigen-decomposition of the Hamiltonian matrix
spctrue, stttrue = jnl.eigh(hmattrue)

# compute propagator matrix
propatrue = stttrue @ jnp.diag(jnp.exp(-1j * spctrue * dt)) @ stttrue.conj().T
np.save(workingdir / 'propatrue', propatrue)
print('propatrue saved.')

# propagate system starting from initial "a" state
# using the Hamiltonian constructed from the true potential
# (used for generating training data)
amattruevec = []
for thisa0 in a0vec:
    tempamat = [thisa0.copy()]
    for _ in range(numts):
        tempamat.append(propatrue @ tempamat[-1])

    amattruevec.append(tempamat)

amattruevec = jnp.array(amattruevec)
np.save(workingdir / 'amattruevec', amattruevec)
print('amattruevec saved.')

print('Done with forward problem.')
print('')  # blank line
# **************************************************