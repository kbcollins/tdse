import numpy as np
import numpy.linalg as nl
import scipy.optimize as so
import scipy.integrate as si
import matplotlib.pyplot as plt


def toeplitz(r, c):
    c = np.asarray(c).ravel()
    r = np.asarray(r).ravel()
    # Form a 1-D array containing a reversed c followed by r[1:] that could be strided to give us toeplitz matrix.
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]
    return np.lib.stride_tricks.as_strided(vals[len(c) - 1:], shape=out_shp, strides=(-n, n)).copy()


# set number of Fourier basis
nfb = 128
# set L of spatial domain
radius = 16
# create vector of real space points for plotting
nx = 1024
xvec = np.linspace(-radius, radius, nx)

# construct matrix to convert Fourier basis coefficients
# into real space values
nvec = np.arange(-nfb, nfb+1)  # nfbvec=-nfb,...,0,...,nfb
convmat = np.exp(1j * np.pi * np.outer(nvec, xvec) / radius) / np.sqrt(2 * radius)


# scaled asymmetric double well potential
def v(x):
    return 0.0003*((x-3)**4 + 10*(x-5)**3)


# compute true potential
vtrue = v(xvec)

# set number of Gaussian basis
ng = 16
# vector of Gaussian basis centers in real space
xg = np.linspace(-radius, radius, ng)


# form initial state in Fourier basis
# rounded box function
def psi0(x):
    return 1.0 + np.tanh((1 - x**2)/0.5)


# normalize psi0
norm = np.sqrt(si.quad(lambda x: np.abs(psi0(x))**2, a=-radius, b=radius)[0])


def normpsi0(x):
    return psi0(x)/norm


# transform psi0 to Fourier basis
vraw = np.zeros(nfb+1, dtype=np.complex128)
for thisn in range(nfb+1):
    def integ(x):
        return (2 * radius) ** (-0.5) * np.exp(-1j * np.pi * thisn * x / radius) * normpsi0(x)

    def rinteg(x):
        return np.real(integ(x))

    def iinteg(x):
        return np.imag(integ(x))
    vraw[thisn] = si.quad(rinteg, a=-radius, b=radius)[0] + 1j * si.quad(iinteg, a=-radius, b=radius)[0]
ainit = np.concatenate([np.conjugate(np.flipud(vraw[1:])), vraw])

# code for propagating system given a potential matrix in the Fourier basis
# and some initial state
# number of elements for toeplitz representation
m = 2*nfb + 1
# set the time step size for propagating
dt = 0.01
# set the number of steps to propagate "a" vector in time
nt = 200
# construct kinetic matrix, this remains constant
kmat = np.diag(np.arange(-nfb, nfb+1) ** 2 * np.pi ** 2 / (2 * radius ** 2))


# propgation function
def propa(thisvmat, thisinita):
    # construct Hamiltonian matrix (in Fourier basis)
    hmatcff = kmat + thisvmat
    # diagonalize the Hamiltonian matrix (eigendecomposition)
    speccff, statescff = np.linalg.eigh(hmatcff)
    # form propagator matrices
    propamat = statescff @ np.diag(np.exp(-1j * speccff * dt)) @ np.conj(statescff.T)
    proplammat = statescff @ np.diag(np.exp(1j * speccff * dt)) @ np.conj(statescff.T)
    # propagate vector, i.e., solve forward problem
    amatcff = np.zeros((nt + 1, 2 * nfb + 1), dtype=np.complex128)
    amatcff[0, :] = np.copy(thisinita)
    for j in range(nt):
        amatcff[j + 1, :] = propamat @ amatcff[j, :]
    return speccff, statescff, amatcff, proplammat


# this code propagates the lambda vector backward in time,
# i.e., solves the adjoint problem
def proplam(thisamat, thisamattrue, proplammat):
    lambmat = np.zeros((nt + 1, 2 * nfb + 1), dtype=np.complex128)
    lambmat[nt, :] = thisamat[nt, :] - thisamattrue[nt, :]
    for j in range(nt - 1, 0, -1):
        lambmat[j, :] = thisamat[j, :] - thisamattrue[j, :] + proplammat @ lambmat[j + 1, :]
    return lambmat


# this function computes the Lagrangian given a set of Gaussian coefficients
def justlag(cffprdt, thisamattrue, thisainit):
    global glbspecprdt, glbstatesprdt, glbamatprdt, glblambmat
    global glbproplammat
    # propagate inita with cffobjecfn
    glbspecprdt, glbstatesprdt, glbamatprdt, glbproplammat = propa(gvmat(cffprdt), thisainit)
    # propagate lambmat with glbamat
    glblambmat = proplam(glbamatprdt, thisamattrue, glbproplammat)
    # compute residual
    resid = glbamatprdt - thisamattrue
    lag = 0.5 * np.real(np.sum(np.conj(resid) * resid))
    return lag


# these functions compute the gradient WFT the Gaussian coefficients
def gradhelp(specprdt, statesprdt):
    alldmat = np.zeros((ng, m, m), dtype=np.complex128)
    expspec = np.exp(-1j * dt * specprdt)
    mask = np.zeros((m, m), dtype=np.complex128)
    for ii in range(m):
        for jj in range(m):
            if np.abs(specprdt[ii] - specprdt[jj]) < 1e-8:
                mask[ii, ii] = expspec[ii]
            else:
                mask[ii, jj] = (expspec[ii] - expspec[jj]) / (-1j * dt * (specprdt[ii] - specprdt[jj]))
    for iii in range(ng):
        thisA = statesprdt.conj().T @ gradgvmat[iii] @ statesprdt
        qmat = thisA * mask
        alldmat[iii, :, :] = -1j * dt * statesprdt @ qmat @ statesprdt.conj().T
    return alldmat


def justgrad(*_):
    global glbspecprdt, glbstatesprdt, glbamatprdt, glblambmat
    global glballdmat, glbderivamat
    glbderivamat = np.zeros((2 * m - 1, m, m), dtype=np.complex128)
    # compute alldmat
    glballdmat = gradhelp(glbspecprdt, glbstatesprdt)
    # compute all entries of the gradient at once
    gradients = np.real(np.einsum('ij,ajk,ik->a', np.conj(glblambmat[1:, :]), glballdmat, glbamatprdt[:-1, :]))
    return gradients


# initialize learned coefficients with uniform random value
cffform = np.random.uniform(size=ng) - 0.5

# loop for evaluating different values of alpha
for alpha in np.linspace(0.06, 0.1836, 30):
    # form matrix for deriving Gaussian basis coefficients
    gmat = np.exp(-alpha * (xg[:, np.newaxis] - xg)**2)
    # form matrix for transforming Gaussian basis coefficients to real space
    gmatplot = np.exp(-alpha * (xvec[:, np.newaxis] - xg)**2)

    # code for computing potential matrix and its gradient
    # from the Gaussian coefficients
    k = np.pi * np.arange(0, 2 * nfb + 1) / radius
    expmat = np.exp((-k ** 2 / (4 * alpha))[:, np.newaxis] + (-1j * k[:, np.newaxis] * xg))
    gvmatcnst = (1 / (2 * radius)) * np.sqrt(np.pi / alpha) * expmat
    # gradient
    gradgvmat = np.zeros((ng, 2 * nfb + 1, 2 * nfb + 1), dtype=np.complex128)
    for i in range(ng):
        gradgvmat[i, :] = toeplitz(gvmatcnst.T[i].conj(), gvmatcnst.T[i])
    # potential matrix

    def gvmat(cff):
        column = gvmatcnst @ cff.astype(np.complex128)
        row = column.conj()
        return toeplitz(r=row, c=column)

    # compute the true Gaussian coefficents from the true potential
    cfftrue = nl.inv(gmat) @ v(xg)

    # computes amattruetrain using inita and the true potential
    spectrue, statestrue, amattrue, _ = propa(gvmat(cfftrue), ainit)
    # transform amattruetrain to real space
    psimattrue = amattrue @ convmat

    # let the user know what's happening
    print(f'Minimizing... alpha={alpha}')
    # compute the predicted coefficients
    rsltform = so.minimize(justlag, cffform, jac=justgrad, method='BFGS',
                           args=(amattrue, ainit),
                           options={'disp': True, 'maxiter': 100}).x

    # use learned coefficients to compute predicted potential
    vformprdc = gmatplot @ rsltform

    # plot result
    trim = 25
    plt.plot(xvec[trim:-trim], vformprdc[trim:-trim], color='red', label='Prediction')
    plt.plot(xvec[trim:-trim], vtrue[trim:-trim], color='black', label='True')
    plt.text(0, 4, f'alpha={alpha}', color='black')
    plt.xlabel('x')
    plt.ylabel('v(x)')
    plt.legend()
    plt.show()
