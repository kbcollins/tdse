import numpy as np
import scipy.linalg as sl
import scipy.special as ss
import scipy.integrate as si
from jax.config import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)


class fourier:

    def __init__(self, L, numx, numfour, theta=None, seed=None):
        #####################################################
        # This method is called when the class object is
        # instantiated.
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #####################################################

        # unpack the model parameters from args

        # store model parameters as class instance variables
        self.L = L
        self.numx = numx
        # this numfour is the number of Fourier basis used
        # to discretize the TDSE PDE
        self.numfour = numfour
        self.numtoepelms = 2 * numfour + 1

        if theta is None:
            if seed is None:
                self.theta = None
            else:
                self.randtheta(dist='uniform', seed=seed)
        else:
            self.theta = theta

        #####################################################
        # The rest of this code is used for creating and
        # storing structures necessary for using the model
        #####################################################

        # vector of Fourier mode indices
        # fournvec = -numfour,...,0,...,numfour
        fournvec = np.arange(-numfour, numfour + 1)

        # real space grid points (for plotting)
        xvec = np.linspace(-L, L, numx)

        # matrix for converting vector Fourier model
        # coefficients to real space, that is, given
        # {\nu_n}_{n=-F}^F = (2L)^{-1/2} \int_{x=-L}^L e^{-i \pi n x / L} fn(x) dx
        # {fnxvec_m}_{m=0}^N = \sum_{n=-F}^F \nu_n e^{i \pi n m \Delta x / L}
        # used like realspacevec = fourspacevec @ fourtox
        self.fourtoxmat = np.exp(1j * np.pi * np.outer(fournvec, xvec) / L) / np.sqrt(2 * L)

        # Toeplitz indexing matrix, used for constructing
        # Toeplitz matrix from a vector which as been set up like:
        # jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
        aa = (-1) * np.arange(0, self.numtoepelms).reshape(self.numtoepelms, 1)
        bb = [np.arange(self.numtoepelms - 1, 2 * self.numtoepelms - 1)]
        self.toepindxmat = np.array(aa + bb)

    def settheta(self, theta):
        self.theta = theta

    def gettheta(self):
        return jnp.array( self.theta )

    def randtheta(self, dist='uniform', seed=None):
        #####################################################
        # This method generates a model specific theta which
        # is filled with random values.
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #####################################################

        if dist=='normal':
            theta = 0.001 * np.random.default_rng(seed).normal(size=2*self.numtoepelms - 1)  # mean=0, std=1, scale=0.001
        elif dist=='uniform':
            theta = 0.02 * np.random.default_rng(seed).random(size=2*self.numtoepelms - 1) - 0.01  # interval=[-0.01, 0.01)
        else:
            print('Error fourier.init(): Distribution selection not recognized.')

        self.theta = theta

    def tox(self):
        ##################################################
        # This method transforms self.theta into a
        # real space potential.
        # - self.theta is the potential
        #   operator matrix in Toeplitz form with its real
        #   and imaginary parts concatenated together.
        ##################################################

        # first we need to transform theta into a complex valued
        # vector
        thetaR = self.theta[:self.numtoepelms]
        thetaI = jnp.concatenate((jnp.array([0.0]), self.theta[self.numtoepelms:]))
        thetaC = thetaR + 1j * thetaI

        # Mathematically,
        # thetaC_j = (2L)^{-1} \int_{x=-L}^L e^{-i \pi j x / L} fn(x) dx; j = {0, ..., 2F + 1}
        # but we want it to be
        # thetaC_n = (2L)^{-1/2} \int_{x=-L}^L e^{-i \pi n x / L} fn(x) dx; n = {-F, ..., F}
        # so we need to multiply (2L)^{1/2} thetaC
        scaledthetaC = np.sqrt(2 * self.L) * thetaC
        # then adjust the elements slightly, because we know the fn we
        # are trying to approximate with theta is real, the imaginary
        # part (n < 0) is just the complex conjugate of the real part (n >= 0)
        recmodelcoeff = np.concatenate([np.conjugate(np.flipud(scaledthetaC[1:(self.numfour + 1)])), scaledthetaC[:(self.numfour + 1)]])
        potentialxvec = jnp.real(recmodelcoeff @ self.fourtoxmat)

        return potentialxvec


    def tovmat(self):
        ##################################################
        # This method transforms self.theta into potential
        # operator matrix vmat in terms of w/e orthonormal
        # basis was used to discretize the TDSE
        # - Here we used the Fourier basis so theta is a
        #   vector containing the concatenation of the
        #   real and imaginary parts of vmat, its size
        #   should be:
        #   2 * numtoepelms - 1 = 4 * numfour + 1
        #################################################

        # to use theta we need to first recombine the real
        # and imaginary parts into a vector of complex values
        vtoephatR = self.theta[:self.numtoepelms]
        vtoephatI = jnp.concatenate((jnp.array([0.0]), self.theta[self.numtoepelms:]))
        vtoephat = vtoephatR + 1j * vtoephatI

        # construct vmathat from complex toeplitz vector
        vmathat = jnp.concatenate([jnp.flipud(jnp.conj(vtoephat)), vtoephat[1:]])[self.toepindxmat]

        return vmathat

    def grad(self):
        ##################################################
        # This method computes \nabla_\theta H(\theta)
        # where H(\theta) = K + V(\theta)
        # Thus, this returns either the gradient of vmat
        # or the model representation of v(x) given
        # self.theta
        ##################################################

        # this code computes the real part of \nabla_\theta H(\theta)
        myeye = jnp.eye(self.numtoepelms)
        wsR = jnp.hstack([jnp.fliplr(myeye), myeye[:,1:]]).T
        ctrmatsR = wsR[self.toepindxmat]

        # this code computes the imaginary part of \nabla_\theta H(\theta)
        wsI = 1.0j * jnp.hstack([-jnp.fliplr(myeye), myeye[:, 1:]])
        wsI = wsI[1:, :]
        wsI = wsI.T
        ctrmatsI = wsI[self.toepindxmat]

        # concatenate the real and imaginary parts of
        gradmat = jnp.concatenate([ctrmatsR, ctrmatsI], axis=2)

        return gradmat

    def fntomodel(self, fn):
        #################################################
        # This method takes a function and returns the
        # model representation of it without using
        # structures internal to the class
        # - Math:
        #   - fn = sum_{j=-F}^F fnmodelcoeff_n * e^{i \pi n x / L}
        #   - fnmodelcoeff_m = <\phi_m, fn>
        #                    = (2L)^{-1/2} \int_{x=-L}^L e^{-i \pi n x / L} fn(x) dx
        # - This can be used to transform (un-normalized) wave
        #   functions to the Fourier representation
        # - fn is a 1D real- or complex-valued function,
        #   that is a callable that takes floating point input
        #   and returns a real- or complex-valued output
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #################################################
        def basisfn(x, n):
            return np.exp(1j * np.pi * n * x / self.L) / np.sqrt(2 * self.L)

        # compute the potential operator matrix, vmat
        fnmodelcoeff = []
        for thisfourn in range(-self.numfour, self.numfour + 1):
            def intgrnd(x):
                return np.conj(basisfn(x,thisfourn)) * fn(x)
            def rintgrnd(x):
                return intgrnd(x).real
            def iintgrnd(x):
                return intgrnd(x).imag
            fnmodelcoeff.append(si.quad(rintgrnd, -self.L, self.L, limit=100)[0] + 1j * si.quad(iintgrnd, -self.L, self.L, limit=100)[0])

        fntoepC = np.array(fnmodelcoeff)
        theta = np.concatenate((fntoepC.real, fntoepC[1:].imag))
        self.theta = theta
        return jnp.array(fntoepC)

class cheby:

    def __init__(self, L, numx, numfour, numcheb, theta=None, seed=None):
        #####################################################
        # This method is called when the class object is
        # instantiated.
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #####################################################

        # unpack the model parameters from args
        L, numx, numfour, numcheb = args

        # store model parameters
        self.L = L
        self.numx = numx
        # this numfour is the number of Fourier basis used
        # to discretize the TDSE PDE
        self.numfour = numfour
        self.numcheb = numcheb

        if theta is None:
            if seed is None:
                self.theta = None
            else:
                self.randtheta(dist='uniform', seed=seed)
        else:
            self.theta = theta

        #####################################################
        # The rest of this code is used for creating and
        # storing structures necessary for using the model
        #####################################################

        # real space grid points (for plotting)
        xvec = np.linspace(-L, L, numx)

        chebnvec = np.arange(0, self._numcheb + 1)

        # matrix for transforming Chebyshev coefficients to
        # real space
        # used like: self._chebtox @ cheb_cff_vec
        self.chebtox = ss.eval_chebyt(np.expand_dims(chebnvec, 0), np.expand_dims(xvec / L, 1))

        # matrix for transforming the Chebyshev representation
        # to Fourier representation (this is used in the adjoint
        # method to construct vhatmat)
        chebtofourmat = []
        for thischebn in range(numcheb + 1):
            temptoeprow = []
            for thisfourn in range(2 * numfour + 1):
                def intgrnd(x):
                    return ss.eval_chebyt(thischebn, x / L) * np.exp(-1j * np.pi * thisfourn * x / L) / (2 * L)
                def rintgrnd(x):
                    return intgrnd(x).real
                def iintgrnd(x):
                    return intgrnd(x).imag
                temptoeprow.append(si.quad(rintgrnd, -L, L, limit=100)[0] + 1j * si.quad(iintgrnd, -L, L, limit=100)[0])
            chebtofourmat.append(sl.toeplitz(r=temptoeprow, c=np.conj(temptoeprow)))

        # used like: self._chebtofourmat @ cheb_cff_vec
        self.chebtofourmat = jnp.array(np.transpose(np.array(chebtofourmat), [1, 2, 0]))

    def settheta(self, theta):
        self.theta = theta

    def gettheta(self):
        return self.theta

    def randtheta(self, dist='uniform', seed=None):
        #####################################################
        # This function generates a model specific theta which
        # is filled with random values.
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #   - the last element in args should always be the number,
        #     of elements/shape of theta
        #####################################################

        # the total number of Chebyshev coefficients = numcheb + 1
        if dist=='normal':
            theta = 0.001 * np.random.default_rng(seed).normal(size=self.numcheb + 1)  # mean=0, std=1
        elif dist=='uniform':
            theta = 10.0 * np.random.default_rng(seed).uniform(size=self.numcheb + 1) - 5.0  # mean=0, interval=[-5.0, 5.0)
        else:
            print('Error fourier.init(): Distribution selection not recognized.')

        self.theta = theta

    def tox(self):
        ##################################################
        # This method transforms self.theta into a
        # real space potential
        ##################################################

        return self.chebtox @ self.theta

    def tovmat(self):
        ##################################################
        # This method transforms self.theta into the
        # potential operator matrix vmat in terms of w/e
        # orthonormal basis was used to discretize the TDSE
        ##################################################

        vmathat = self.chebtofourmat @ self.theta
        return vmathat

    def grad(self):
        ##################################################
        # This method computes \nabla_\theta H(\theta)
        # where H(\theta) = K + V(\theta)
        # Thus, this returns either the gradient of vmat
        # or the model representation of v(x) given
        # self.theta
        ##################################################

        gradmat = self.chebtofourmat

        return gradmat

    def fntomodel(self, fn):
        #################################################
        # This method takes a function and returns
        # the theta
        # - theta is the structure given to the optimizer
        #   for learning and may be different from the
        #   model representation of a function (e.g.,
        #   the Fourier model represents a function as
        #   a set of complex valued coefficients, but we
        #   give the optimizer a concatenation of the real
        #   and imaginary parts of the model)
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #################################################

        kvec = np.arange(1, self.numcheb + 2)
        chebnvec = np.arange(0, self.numcheb + 1)

        chebweights = np.ones(self.numcheb + 1)
        chebweights[0] = 0.5

        def chebtheta(k):
            return (k - 0.5) * np.pi / (self.numcheb + 1)

        def g(k):
            return fn(L * np.cos(chebtheta(k)))

        chebvec = 2 * np.sum(g(kvec) * np.cos(chebnvec[..., np.newaxis] * chebtheta(kvec)), axis=1) / (self.numcheb + 1)

        chebvec = chebweights * chebvec

        self.theta = chebvec

    # alias fntotheta to fntocheby because they are the same thing
    fntotheta = fntomodel
