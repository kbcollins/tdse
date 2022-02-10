from time import time_ns
import numpy as np
import numpy.linalg as nl
import scipy.optimize as so
import scipy.integrate as si
import matplotlib.pyplot as plt
from numba import njit


@njit
def toeplitz(r, c):
    c = np.asarray(c).ravel()
    r = np.asarray(r).ravel()
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]
    return np.lib.stride_tricks.as_strided(vals[len(c)-1:], shape=out_shp, strides=(-n, n)).copy()


# define potential for training data
def v(x, choice=0):
    if choice == 0:
        # dimensionless quantum harmonic oscillator potential
        return 0.5 * x**2
    elif choice == 1:
        # symmetric double well potential
        return 0.0025 * (x ** 2 - 25) ** 2
    elif choice == 2:
        # asymmetric double well potential
        return 0.0003 * ((x - 3) ** 4 + 10 * (x - 5) ** 3)
    elif choice == 3:
        # soft coulomb potential
        return -1 / np.sqrt(x ** 2 + 0.25)
    else:
        print('Function v(x, choice=0): Did not recognise your input for choice.')


# define system's initial wave function,
def gen_inita(bounda, boundb, choice=0):
    def psi0(x):
        if choice == 0:
            # rounded box function
            return 1.0 + np.tanh((1 - x**2)/0.5)
        elif choice == 1:
            # triangular pulse
            return np.piecewise(x, [x < -1, (x >= -1) & (x <= 1), x > 1], [0, lambda z: (1 - np.abs(z)), 0])
        elif choice == 2:
            # parabolic pulse
            return np.piecewise(x, [x < -1, (x >= -1) & (x <= 1), x > 1], [0, lambda z: (1 - z**2), 0])
        elif choice == 3:
            # hyperbolic secant squared
            return (np.cosh(x))**(-2)
        elif choice == 4:
            # Laplace
            return np.exp(-np.abs(x))
        else:
            print('Function psi0(x): Did not recognise your input for choice.')
            return None

    # normalize psi0
    norm = np.sqrt(si.quad(lambda x: np.abs(psi0(x))**2, a=bounda, b=boundb)[0])
    def normpsi0(x):
        return psi0(x)/norm

    # transform to Fourier basis
    vraw = np.zeros(nfb+1, dtype=np.complex128)
    for thisn in range(nfb+1):
        def integ(x):
            return (2 * radius) ** (-0.5) * np.exp(-1j * np.pi * thisn * x / radius) * normpsi0(x)
        def rinteg(x):
            return np.real(integ(x))
        def iinteg(x):
            return np.imag(integ(x))
        vraw[thisn] = si.quad(rinteg, a=bounda, b=boundb)[0] + 1j * si.quad(iinteg, a=bounda, b=boundb)[0]

    return np.concatenate([np.conjugate(np.flipud(vraw[1:])), vraw])


print('Initializing computational variables.')

# set number of Fourier basis
nfb = 128
# set L of spatial domain
radius = 16
# create vector of real space points for plotting
nx = 1024
xvec = np.linspace(-radius, radius, nx)
# matrix to convert Fourier basis coefficients
# into real space values
nvec = np.arange(-nfb, nfb+1) # nfbvec=-nfb,...,0,...,nfb
convmat = np.exp(1j * np.pi * np.outer(nvec, xvec) / radius) / np.sqrt(2 * radius)
# compute true potential
vchoice = 2
vtrue = v(x=xvec, choice=vchoice)
# generate initial a state
inita = gen_inita(bounda=-radius, boundb=radius, choice=0)

# initialize variables to track
ngvec = []
comptime = []
condnum = []
rndtheta = []
rndvtrue = []
rndvtrain = []
rndprop = []

for i in range(5):
    # set number of Gaussian basis
    ng = 16 + 2**(2 * i)
    ngvec.append(ng)

    # vector of Gaussian basis centers in real space
    xg, dg = np.linspace(-radius, radius, ng, retstep=True)
    # alpha sets the amount of overlap for Gaussian basis
    # here we determine alpha by setting the full width half max
    # equal to the spacing between Gaussian centers
    alpha = 4 * np.log(2) / dg**2
    gmat = np.exp(-alpha * (xg[:,np.newaxis] - xg)**2)
    condnum.append(nl.cond(gmat))
    # matrix to convert real space to Gaussian basis
    gbreal2guass = nl.inv(gmat)
    # matrix to convert Gaussian basis to real space
    gbguass2real = np.exp(-alpha * (xvec[:,np.newaxis] - xg)**2)
    # compute Gaussian basis coefficients for true potential
    gbcfftrue = gbreal2guass @ v(xg, choice=vchoice)
    # reconstruct the true potential from the Gaussian basis
    vtruerecon = gbguass2real @ gbcfftrue

    # define function for generating the potential matrix
    # in the Fourier basis from Gaussian basis coefficients
    k = np.pi * np.arange(0, 2*nfb+1) / radius
    expmat = np.exp((-k**2 / (4*alpha))[:, np.newaxis] + (-1j*k[:,np.newaxis]*xg))
    cnstgvmat = (1 / (2 * radius)) * np.sqrt(np.pi / alpha) * expmat


    @njit
    def gbvmat(cff):
        column = cnstgvmat @ cff.astype(np.complex128)
        row = column.conj()
        return toeplitz(r=row, c=column)


    # compute gradient of Gaussian basis potential matrix
    gradgbvmat = np.zeros((ng, 2 * nfb + 1, 2 * nfb + 1), dtype=np.complex128)
    for i in range(ng):
        gradgbvmat[i, :]=toeplitz(cnstgvmat.T[i].conj(), cnstgvmat.T[i])

    # set number of elements for toeplitz representation
    m = 2 * nfb + 1
    # set the time step size
    dt = 0.01
    # set the number of time steps to propagate
    nt = 200
    # compute kinetic matrix (this is constant for given system)
    kmat = np.diag( np.arange(-nfb, nfb+1)**2 * np.pi**2 / (2 * radius**2) )


    # define function for propagating system
    @njit
    def propa(gbcff, initapropa, dtpropa, ntpropa=200):
        # form Fourier basis potential matrix
        vmatpropa = gbvmat(gbcff)
        # construct Hamiltonian matrix (in Fourier basis)
        hmatpropa = kmat + vmatpropa
        # eigen-decomposition Hamiltonian matrix
        specpropa, statespropa = np.linalg.eigh(hmatpropa)
        # form a vector propagator matrix
        propamat = statespropa @ np.diag(np.exp(-1j * specpropa * dtpropa)) @ statespropa.T.conj()
        # propagate a vector
        amat = np.zeros((ntpropa + 1, 2 * nfb + 1), dtype=np.complex128)
        amat[0, :] = np.copy(initapropa)
        for j in range(ntpropa):
            amat[j + 1, :] = propamat @ amat[j, :]
        return specpropa, statespropa, amat


    # define function for propagating lambda
    @njit
    def proplam(specproplam, statesproplam, amat, amattrueproplam, dtproplam, ntproplam=200):
        # form lambda vector propagator matrix
        proplammat = statesproplam @ np.diag(np.exp(1j * specproplam * dtproplam)) @ statesproplam.T.conj()
        # propagate lambda vector
        lammat = np.zeros((ntproplam + 1, 2 * nfb + 1), dtype=np.complex128)
        lammat[ntproplam, :] = amat[ntproplam, :] - amattrueproplam[ntproplam, :]
        for j in range(ntproplam - 1, 0, -1):
            lammat[j, :] = amat[j, :] - amattrueproplam[j, :] + proplammat @ lammat[j + 1, :]
        return lammat


    # compute the true propagation from the true potential
    # to use as training data
    print('Computing training data.')
    _, _, amattruetrain = propa(gbcff=gbcfftrue, initapropa=inita, dtpropa=dt, ntpropa=nt)


    # define objective function
    def objecfn(cffobjecfn):
        global glbobjhis, glbitrcnt
        global glbspec, glbstates, glbamat, glblammat
        # propagate initial state with cffobjecfn
        glbspec, glbstates, glbamat = propa(gbcff=cffobjecfn,
                                            initapropa=inita,
                                            dtpropa=dt,
                                            ntpropa=nt)
        # propagate lambda with glbamat
        glblammat = proplam(specproplam=glbspec,
                            statesproplam=glbstates,
                            amat=glbamat,
                            amattrueproplam=amattruetrain,
                            dtproplam=dt,
                            ntproplam=nt)
        # compute objective
        resid = glbamat - amattruetrain
        objec = 0.5 * np.real( np.sum(np.conj(resid) * resid) )
        glbitercnt += 1
        # put the newest value of objective on the end of the stack
        glbobjechist = np.roll(glbobjechist, -1)
        glbobjechist[-1] = objec
        # print latest values of the objective
        print(f'\x1b[1K\r{glbitercnt} Objective={glbobjechist[-1]}', end='')
        return objec


    # define helper function for gradfn
    @njit
    def gradfnhelp(dtgradfnhelp, gradgbvmatgradfnhelp, mgradfnhelp, nggradfnhelp, specgradfnhelp, statesgradfnhelp):
        dmat = np.zeros((nggradfnhelp, mgradfnhelp, mgradfnhelp), dtype=np.complex128)
        expspec = np.exp(-1j * dtgradfnhelp * specgradfnhelp)
        mask = np.zeros((mgradfnhelp, mgradfnhelp), dtype=np.complex128)
        for ii in range(mgradfnhelp):
            for jj in range(mgradfnhelp):
                if np.abs(specgradfnhelp[ii] - specgradfnhelp[jj]) < 1e-8:
                    mask[ii, ii] = expspec[ii]
                else:
                    mask[ii, jj] = (expspec[ii] - expspec[jj]) / (-1j * dtgradfnhelp * (specgradfnhelp[ii] - specgradfnhelp[jj]))
        for iii in range(nggradfnhelp):
            thisA = statesgradfnhelp.conj().T @ gradgbvmatgradfnhelp[iii] @ statesgradfnhelp
            qmat = thisA * mask
            dmat[iii, :, :] = -1j * dtgradfnhelp * statesgradfnhelp @ qmat @ statesgradfnhelp.conj().T
        return dmat


    # define gradient function
    def gradfn(_):
        global glbspec, glbstates, glbamat, glblammat, glbdmat
        # compute dmat
        glbdmat = gradfnhelp(dtgradfnhelp=dt,
                             gradgbvmatgradfnhelp=gradgbvmat,
                             mgradfnhelp=m,
                             nggradfnhelp=ng,
                             specgradfnhelp=glbspec,
                             statesgradfnhelp=glbstates)
        # compute all entries of the gradient at once
        gradients = np.real(np.einsum('ij,ajk,ik->a', np.conj(glblammat[1:, :]), glbdmat, glbamat[:-1, :]))
        return gradients


    # initialize model parameters to uniform random values
    print('Initializing training parameters.')
    seed = 1272022  # set to None for random initialization each time
    cffform = np.random.default_rng(seed).uniform(low=-0.5, high=0.5, size=ng) - 0.5

    # set the size of the objective's history
    objechistlen = 50
    # create the objective's history
    glbobjhis = np.zeros(objechistlen)
    # initialize variable for counting the number of iterations
    glbitrcnt = 0

    print('Optimizing.')
    # get system time before optimizing
    thistime = time_ns()
    # use minimize to learn prediction for potential
    resform = so.minimize(objecfn, cffform, jac=gradfn, method='BFGS',
                          options={'disp': True, 'maxiter': 100, }).x
    # get system time after optimizing, take difference and store value
    # difference will be in nanoseconds
    comptime.append( (time_ns() - thistime) / 1e9 )
    print(f'Optimization took {comptime[-1]} seconds.')

    # plot the objective's history
    plt.plot(glbobjhis)
    plt.title(f'Last {objechistlen} Values of the Objective. NGB={ng}')
    plt.ylabel('Magnitude')
    # plt.show()
    plt.savefig(f'./tdse-adj-kbc-gauss-script-results/objective{ng}.pdf')
    plt.clf()

    # results
    # Relative norm difference of predicted theta to true theta
    rndtheta.append(nl.norm(resform - gbcfftrue) / nl.norm(gbcfftrue))
    # use the predicted theta to compute the predicted potential
    vformprdc = gbguass2real @ resform
    # Relative norm difference of the predicted potential to true potential
    rndvtrue.append(nl.norm(vtrue - vformprdc) / nl.norm(vtrue))
    # Relative norm difference of the predicted potential to training data
    rndvtrain.append(nl.norm(vtruerecon - vformprdc) / nl.norm(vtruerecon))

    # plot potentials
    trim = 1
    plt.plot(xvec[trim:-trim], vformprdc[trim:-trim], color='red', label='Learned')
    plt.plot(xvec[trim:-trim], vtruerecon[trim:-trim], color='blue', label='Training')
    plt.plot(xvec[trim:-trim], vtrue[trim:-trim], color='black', label='True')
    plt.title(f'Potentials. NGB={ng}')
    plt.xlabel('Position')
    plt.ylabel('Magnitude of Potential')
    plt.legend()
    # plt.show()
    plt.savefig(f'./tdse-adj-kbc-gauss-script-results/potentials{ng}.pdf')
    plt.clf()

    # test propagation
    # set how far past the training data to propagate
    ntextra=500
    # compute the true propagation from the true potential
    _, _, amattruetest = propa(gbcff=gbcfftrue, initapropa=inita, dtpropa=dt, ntpropa=nt+ntextra)
    # transform amattruetest to real space
    psimattruetest = amattruetest @ convmat
    # propagate with predicted potential
    _, _, amatformprdc = propa(gbcff=resform, initapropa=inita, dtpropa=dt, ntpropa=nt+ntextra)
    # transform amatformprdc to real space
    psimatformprdc = amatformprdc @ convmat
    # relative norm difference of the predicted wave function
    # to true wave function
    rndprop.append(nl.norm(psimatformprdc - psimattruetest) / nl.norm(psimattruetest))
    # plot the probability density of the final state
    plt.plot(xvec, np.abs(psimatformprdc[-1]) ** 2, 'r', label='Predicted')
    plt.plot(xvec, np.abs(psimattruetest[-1]) ** 2, 'ksvec', label='Test')
    plt.legend()
    plt.title(f'Probability Density of the Final State. NGB={ng}')
    plt.xlabel('Position')
    plt.ylabel('Probability')
    # plt.show()
    plt.savefig(f'./tdse-adj-kbc-gauss-script-results/prop{ng}.pdf')
    plt.clf()

# plot computational time as a function of the number of basis
plt.plot(ngvec, comptime)
plt.title('Computational Time as a Function of Basis')
plt.xlabel('Number of Basis')
plt.ylabel('Time (sec)')
# plt.show()
plt.savefig('./tdse-adj-kbc-gauss-script-results/comp-time.pdf')
plt.clf()

# plot condition number as a function of the number of basis
plt.plot(ngvec, condnum)
plt.title('Condition Number as a Function of Basis')
plt.xlabel('Number of Basis')
plt.ylabel('Magnitude')
# plt.show()
plt.savefig('./tdse-adj-kbc-gauss-script-results/condition.pdf')
plt.clf()

# plot relative norm difference of theta
plt.plot(ngvec, rndtheta)
plt.title('Relative Norm Difference of Theta as a Function of Basis')
plt.xlabel('Number of Basis')
plt.ylabel('Magnitude')
# plt.show()
plt.savefig('./tdse-adj-kbc-gauss-script-results/diff-theta.pdf')
plt.clf()

# plot relative norm difference of potential WRT truth
plt.plot(ngvec, rndvtrue)
plt.title('Relative Norm Difference of the Potential WRT Truth as a Function of Basis')
plt.xlabel('Number of Basis')
plt.ylabel('Magnitude')
# plt.show()
plt.savefig('./tdse-adj-kbc-gauss-script-results/diff-truth.pdf')
plt.clf()

# plot relative norm difference of potential WRT training potential
plt.plot(ngvec, rndvtrain)
plt.title('Relative Norm Difference of the Potential WRT Training as a Function of Basis')
plt.xlabel('Number of Basis')
plt.ylabel('Magnitude')
# plt.show()
plt.savefig('./tdse-adj-kbc-gauss-script-results/diff-train.pdf')
plt.clf()

# plot relative norm difference of the propagated wave function
# WRT truth
plt.plot(ngvec, rndprop)
plt.title('Relative Norm Difference of Propagation WRT Truth as a Function of Basis')
plt.xlabel('Number of Basis')
plt.ylabel('Magnitude')
# plt.show()
plt.savefig('./tdse-adj-kbc-gauss-script-results/diff-prop.pdf')
plt.clf()
