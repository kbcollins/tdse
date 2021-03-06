import numpy as np
import scipy.linalg as sl
import scipy.special as ss
import scipy.integrate as si
from jax.config import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)


class fourier:

    def __init__(self, *args, seed=None):
        #####################################################
        # This method is called when the class object is
        # instantiated.
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #####################################################

        # unpack the model parameters from args
        L, numx, numfour = args

        # store model parameters as class instance variables
        self._L = L
        self._numx = numx
        # this numfour is the number of Fourier basis used
        # to discretize the TDSE PDE
        self._numfour = numfour

        # create and initialize theta with random values
        # - theta is the model representation of the potential
        # - all model objects have the class method self.randtheta
        #   to generate a theta of random values, the result is
        #   stored as the instance variable self.theta
        self.theta = self.randtheta(self._L, self._numx, self._numfour, dist='uniform', seed=seed)


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
        self._fourtoxmat = np.exp(1j * np.pi * np.outer(fournvec, xvec) / L) / np.sqrt(2 * L)

        self._numtoepelms = 2 * numfour + 1

        # Toeplitz indexing matrix, used for constructing
        # Toeplitz matrix from a vector which as been set up like:
        # jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
        aa = (-1) * np.arange(0, self._numtoepelms).reshape(self._numtoepelms, 1)
        bb = [np.arange(self._numtoepelms - 1, 2 * self._numtoepelms - 1)]
        self._toepindxmat = np.array(aa + bb)


    def randtheta(*args, dist='uniform', seed=None):
        #####################################################
        # This method generates a model specific theta which
        # is filled with random values.
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #####################################################

        # the last argument passed should always be numfour,
        # the rest can be ignored for this function
        numfour = args[-1]
        numtoepelms = 2 * numfour + 1

        if dist=='normal':
            theta = 0.001 * np.random.default_rng(seed).normal(size=numtoepelms * 2 - 1)  # mean=0, std=1, scale=0.001
        elif dist=='uniform':
            theta = 0.02 * np.random.default_rng(seed).random(size=2 * numtoepelms - 1) - 0.01  # interval=[-0.01, 0.01)
        else:
            print('Error fourier.init(): Distribution selection not recognized.')

        return theta


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
        thetaR = self.theta[:self._numtoepelms]
        thetaI = jnp.concatenate((jnp.array([0.0]), self.theta[self._numtoepelms:]))
        thetaC = thetaR + 1j * thetaI

        # Mathematically,
        # thetaC_j = (2L)^{-1} \int_{x=-L}^L e^{-i \pi j x / L} fn(x) dx; j = {0, ..., 2F + 1}
        # but we want it to be
        # thetaC_n = (2L)^{-1/2} \int_{x=-L}^L e^{-i \pi n x / L} fn(x) dx; n = {-F, ..., F}
        # so we need to multiply (2L)^{1/2} thetaC
        scaledthetaC = np.sqrt(2 * self._L) * thetaC
        # then adjust the elements slightly, because we know the fn we
        # are trying to approximate with theta is real, the imaginary
        # part (n < 0) is just the complex conjugate of the real part (n >= 0)
        recmodelcoeff = np.concatenate([np.conjugate(np.flipud(scaledthetaC[1:(self._numfour + 1)])), scaledthetaC[:(self._numfour + 1)]])

        potentialxvec = jnp.real(recmodelcoeff @ self._fourtoxmat)

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
        vtoephatR = self.theta[:self._numtoepelms]
        vtoephatI = jnp.concatenate((jnp.array([0.0]), self.theta[self._numtoepelms:]))
        vtoephat = vtoephatR + 1j * vtoephatI

        # construct vmathat from complex toeplitz vector
        vmathat = jnp.concatenate([jnp.flipud(jnp.conj(vtoephat)), vtoephat[1:]])[self._toepindxmat]

        return vmathat


    def thetatovmat(theta, *args):
        #################################################
        # This method transforms theta (passed to the
        # method as an argument) into the potential
        # operator matrix vmat in terms of w/e orthonormal
        # basis was used to discretize the TDSE
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        # - theta is a vector containing the concatenation
        #   of the real and imaginary parts of vmat
        #   its size should be
        #   2 * numtoepelms - 1 = 4 * numfour + 1
        #################################################

        # unpack the model parameters from args
        L, numx, numfour = args
        numtoepelms = 2 * numfour + 1

        # Toeplitz indexing matrix, used for constructing
        # Toeplitz matrix from a vector which as been set up like:
        # jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
        aa = (-1) * np.arange(0, numtoepelms).reshape(numtoepelms, 1)
        bb = [np.arange(numtoepelms - 1, 2 * numtoepelms - 1)]
        toepindxmat = np.array(aa + bb)

        # to use theta we need to first recombine the real
        # and imaginary parts into a vector of complex values
        vtoephatR = theta[:numtoepelms]
        vtoephatI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
        vtoephat = vtoephatR + 1j * vtoephatI

        # construct vmathat from complex toeplitz vector
        vmathat = jnp.concatenate([jnp.flipud(jnp.conj(vtoephat)), vtoephat[1:]])[toepindxmat]

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
        myeye = jnp.eye(self._numtoepelms)
        wsR = jnp.hstack([jnp.fliplr(myeye), myeye[:,1:]]).T
        ctrmatsR = wsR[self._toepindxmat]

        # this code computes the imaginary part of \nabla_\theta H(\theta)
        wsI = 1.0j * jnp.hstack([-jnp.fliplr(myeye), myeye[:, 1:]])
        wsI = wsI[1:, :]
        wsI = wsI.T
        ctrmatsI = wsI[self._toepindxmat]

        # concatenate the real and imaginary parts of
        gradmat = jnp.concatenate([ctrmatsR, ctrmatsI], axis=2)

        return gradmat


    def thetatograd(theta, *args):
        ##################################################
        # This function computes \nabla_\theta H(\theta)
        # for a given theta without having access to
        # internal structures of the class
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        # - For the Fourier model, the gradient we need is
        #   \nabla_\theta vhatmat which is a constant so
        #   theta isn't used here
        ##################################################

        # unpack the model parameters from args
        L, numx, numfour = args
        numtoepelms = 2 * numfour + 1

        # Toeplitz indexing matrix, used for constructing
        # Toeplitz matrix from a vector which as been set up like:
        # jnp.concatenate([jnp.flipud(row.conj()), row[1:]])
        aa = (-1) * np.arange(0, numtoepelms).reshape(numtoepelms, 1)
        bb = [np.arange(numtoepelms - 1, 2 * numtoepelms - 1)]
        toepindxmat = np.array(aa + bb)

        # this code computes the real part of \nabla_\theta H(\theta)
        myeye = jnp.eye(numtoepelms)
        wsR = jnp.hstack([jnp.fliplr(myeye), myeye[:,1:]]).T
        ctrmatsR = wsR[toepindxmat]

        # this code computes the imaginary part of \nabla_\theta H(\theta)
        wsI = 1.0j * jnp.hstack([-jnp.fliplr(myeye), myeye[:, 1:]])
        wsI = wsI[1:, :]
        wsI = wsI.T
        ctrmatsI = wsI[toepindxmat]

        # concatenate the real and imaginary parts of
        gradmat = jnp.concatenate([ctrmatsR, ctrmatsI], axis=2)

        return gradmat


    def fntomodel(fn, *args):
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
        L, numx, numfour = args

        def basisfn(x):
            return np.exp(1j * np.pi * thisfourn * x / L) / np.sqrt(2 * L)

        # compute the potential operator matrix, vmat
        fnmodelcoeff = []
        for thisfourn in range(-numfour, numfour + 1):
            def intgrnd(x):
                return np.conj(basisfn(x)) * fn(x)
            def rintgrnd(x):
                return intgrnd(x).real
            def iintgrnd(x):
                return intgrnd(x).imag
            fnmodelcoeff.append(si.quad(rintgrnd, -L, L, limit=100)[0] + 1j * si.quad(iintgrnd, -L, L, limit=100)[0])

        fnmodelcoeff = np.array(fnmodelcoeff)

        return fnmodelcoeff


    def fntotheta(fn, *args):
        #################################################
        # This method takes a function and returns
        # the theta, i.e., the structure given to the
        # optimizer for learning, without using any
        # structures internal to the class.
        # - Math:
        #   - vmat_{m n} = <\phi_m, fn \phi_n>
        #                = (2L)^{-1} \int_{x=-L}^L e^{i \pi (n-m) x / L} fn(x) dx
        #           m,n = {-F, ..., 0, ..., F}
        #   - vmattoepC_{j=m-n} = (2L)^{-1} \int_{x=-L}^L e^{i \pi j x / L} fn(x) dx
        #           j = {0, ..., 2 * F + 1}
        #   - vmattoepR = Real(vmattoepC)
        #   - vmattoepI = Imaginary(vmattoepC)
        #   - theta = Concatenate([vmattoepR, vmattoepI])
        # - Theta may be different from the
        #   model representation of a function.
        #   - For example, the Fourier model represents a
        #     function is a set of complex valued
        #     coefficients, but theta is a vector of the
        #     real and imaginary parts of the Toeplitz form
        #     of the potential operator matrix concatenation
        #     together.
        # - fn is assumed to be a 1D real-valued function,
        #   that is a callable that takes a floating point
        #   value as input and returns a floating point value
        #   as output
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #################################################
        L, numx, numfour = args
        numtoepelms = 2 * numfour + 1

        def innerprodbasis(x):
            return np.exp(1j * np.pi * thisfourn * x / L) / (2 * L)

        # compute the potential operator matrix, vmat,
        # in the Toeplitz form
        fntoepC = []
        for thisfourn in range(numtoepelms):
            def intgrnd(x):
                return np.conj(innerprodbasis(x)) * fn(x)
            def rintgrnd(x):
                return intgrnd(x).real
            def iintgrnd(x):
                return intgrnd(x).imag
            fntoepC.append(si.quad(rintgrnd, -L, L, limit=100)[0] + 1j * si.quad(iintgrnd, -L, L, limit=100)[0])

        fntoepC = np.array(fntoepC)

        # concatenate the real and imaginary parts together
        theta = jnp.concatenate((fntoepC.real, fntoepC[1:].imag))

        return theta


class cheby:

    def __init__(self, *args, seed=None):
        #####################################################
        # This method is called when the class object is
        # instantiated.
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #####################################################

        # unpack the model parameters from args
        L, numx, numfour, numcheb = args

        # store model parameters
        self._L = L
        self._numx = numx
        # this numfour is the number of Fourier basis used
        # to discretize the TDSE PDE
        self._numfour = numfour
        self._numcheb = numcheb

        self.theta = self.randtheta(self._L, self._numx, self._numfour, self._numcheb, dist='uniform', seed=seed)


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
        self._chebtox = ss.eval_chebyt(np.expand_dims(chebnvec, 0), np.expand_dims(xvec / L, 1))

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
        self._chebtofourmat = jnp.array(np.transpose(np.array(chebtofourmat), [1, 2, 0]))


    def randtheta(*args, dist='uniform', seed=None):
        #####################################################
        # This function generates a model specific theta which
        # is filled with random values.
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #   - the last element in args should always be the number,
        #     of elements/shape of theta
        #####################################################

        # unpack shape of theta
        numcheb = args[-1]

        # the total number of Chebyshev coefficients = numcheb + 1
        if dist=='normal':
            theta = 0.001 * np.random.default_rng(seed).normal(size=numcheb + 1)  # mean=0, std=1
        elif dist=='uniform':
            theta = 10.0 * np.random.default_rng(seed).uniform(size=numcheb + 1) - 5.0  # mean=0, interval=[-5.0, 5.0)
        else:
            print('Error fourier.init(): Distribution selection not recognized.')

        return theta


    def tox(self):
        ##################################################
        # This method transforms self.theta into a
        # real space potential
        ##################################################

        return self._chebtox @ self.theta


    def tovmat(self):
        ##################################################
        # This method transforms self.theta into the
        # potential operator matrix vmat in terms of w/e
        # orthonormal basis was used to discretize the TDSE
        ##################################################

        vmathat = self._chebtofourmat @ self.theta
        return vmathat


    def thetatovmat(theta, *args):
        ##################################################
        # This method transforms theta (passed to the
        # method as an argument) into the potential
        # operator matrix vmat in terms of w/e orthonormal
        # basis was used to discretize the TDSE
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        ##################################################

        # unpack the model parameters from args
        L, numx, numfour, numcheb = args

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

        # used like: chebtofourmat @ cheb_cff_vec
        chebtofourmat = jnp.array(np.transpose(np.array(chebtofourmat), [1, 2, 0]))

        vmathat = chebtofourmat @ theta

        return vmathat


    def grad(self):
        ##################################################
        # This method computes \nabla_\theta H(\theta)
        # where H(\theta) = K + V(\theta)
        # Thus, this returns either the gradient of vmat
        # or the model representation of v(x) given
        # self.theta
        ##################################################

        gradmat = self._chebtofourmat

        return gradmat


    def thetatograd(_, *args):
        ##################################################
        # This function computes \nabla_\theta H(\theta)
        # for a given theta without using the structures
        # internal to the class
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        ##################################################

        # unpack the model parameters from args
        L, numx, numfour, numcheb = args

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

        # used like: gradmat @ cheb_cff_vec
        gradmat = jnp.array(np.transpose(np.array(chebtofourmat), [1, 2, 0]))

        return gradmat


    def fntomodel(fn, *args):
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

        # unpack the model parameters from args
        L, numx, numfour, numcheb = args

        kvec = np.arange(1, numcheb + 2)
        chebnvec = np.arange(0, numcheb + 1)

        chebweights = np.ones(numcheb + 1)
        chebweights[0] = 0.5

        def chebtheta(k):
            return (k - 0.5) * np.pi / (numcheb + 1)

        def g(k):
            return fn(L * np.cos(chebtheta(k)))

        chebvec = 2 * np.sum(g(kvec) * np.cos(chebnvec[..., np.newaxis] * chebtheta(kvec)), axis=1) / (numcheb + 1)

        chebvec = chebweights * chebvec

        return chebvec


    # alias fntotheta to fntocheby because they are the same thing
    fntotheta = fntomodel
