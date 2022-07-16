#####################################################
# This commented out block of code serves as a template
# for writing new models to use as the potential in
# the adjoint method
#####################################################
class model:

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
        thetashape = args[-1]

        return theta


    def tox(self):
        ##################################################
        # This method transforms self.theta into a
        # real space potential
        ##################################################

        return potentialxvec


    def tovmat(self):
        ##################################################
        # This method transforms self.theta into the
        # potential operator matrix vmat in terms of w/e
        # orthonormal basis was used to discretize the TDSE
        ##################################################

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
        L, numx, numfour = args

        return vmathat


    def grad(self):
        ##################################################
        # This method computes \nabla_\theta H(\theta)
        # where H(\theta) = K + V(\theta)
        # Thus, this returns either the gradient of vmat
        # or the model representation of v(x) given
        # self.theta
        ##################################################

        return gradmat


    def thetatograd(theta, *args):
        ##################################################
        # This function computes \nabla_\theta H(\theta)
        # for a given theta without using the structures
        # internal to the class
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        ##################################################

        # unpack the model parameters from args
        L, numx, numfour = args

        return gradmat


    def fntomodel(fn, *args):
        #################################################
        # This method takes a function and returns the
        # model representation of it without using
        # structures internal to the class
        # - args is a tuple containing the parameters needed
        #   to fully define the mode
        #################################################

        # unpack the model parameters from args
        L, numx, numfour = args


    def fntotheta(fn, *args):
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
        L, numx, numfour = args

        return theta