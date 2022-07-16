import numpy as np
import jax.numpy.linalg as jnl
import tdsemodelclass

###############################################################
# select model of the potential
###############################################################

model = tdsemodelclass.fourier
# model = tdsemodelclass.cheby


###############################################################
# specify the model parameters
# - When the model is fourier modelprms = (L, numx, numfour),
#   when the model is cheby modelprms = (L, numx, numfour, numcheb)
# - I have found from experience the cheby model
#   works best when numcheb is an odd number, such as numcheb = 11
###############################################################


# this is initialized as an empty list, it should be assigned
# a tuple containing the parameters which fully define a model
# when used. Assignment is done like:
# tdseadjoint.modelprms = (L, numx, numfour)
modelprms = []


###############################################################
# make kinetic energy operator matrix, kmat
# - in the Fourier representation, this is constant for a given
#   number of Fourier basis functions (i.e., numfour)
###############################################################

# Function for constructing kmat. This must be run at least once
def mk_kmat(L, numfour):
    global kmat
    kmat = np.diag(np.arange(-numfour, numfour + 1) ** 2 * np.pi ** 2 / (2 * L ** 2))


###############################################################
# forward propagation - make training data
###############################################################

def forward_prop(theta):
    # construct vmathat using the model class method
    # .tovmat(), theta is what ever the model class
    # uses as the data structure to store the potential
    # all other arguments are required to define
    # the model
    vhatmat = model.thetatovmat(theta, *modelprms)

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # **************************************************
    # Construct Hamiltonian matrix
    hhatmat = kmat + vhatmat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hhatmat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T

    # forward propagation loop
    ahatmatvec = []
    for r in range(a0vec.shape[0]):
        thisahatmat = [a0vec[r].copy()]

        # propagate system starting from initial "a" state
        for _ in range(numts):
            # propagate the system one time-step
            thisahatmat.append(propahat @ thisahatmat[-1])

        ahatmatvec.append(thisahatmat)

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



def waveobject(theta):

    # construct vhatmat from theta using the model
    # class method model.thetatovmat
    vhatmat = model.thetatovmat(theta, *modelprms)

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # **************************************************
    # Construct Hamiltonian matrix
    hhatmat = kmat + vhatmat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hhatmat)

    # compute propagator matrix
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T

    # forward propagation loop
    ahatmatvec = []
    for r in range(a0vec.shape[0]):
        thisahatmat = [a0vec[r].copy()]

        # propagate system starting from initial "a" state
        for _ in range(numts):
            # propagate the system one time-step
            thisahatmat.append(propahat @ thisahatmat[-1])

        ahatmatvec.append(thisahatmat)

    # transform python list to a jax.numpy array
    ahatmatvec = jnp.array(ahatmatvec)

    # compute objective functions
    resid = ahatmatvec - amattruevec

    # as per our math, we need to take the real part of
    # the objective
    rtnobjvec = jnp.real(0.5 * jnp.sum(jnp.conj(resid) * resid, axis=1))
    rtnobj = jnp.sum(rtnobjvec)
    # **************************************************

    return rtnobj


# jit wavefnobject()
jitfourwaveobject = jax.jit(fourwaveobject)

# precompile jitwavefnobject with Chebyshev representation
# of the true potential
# JAX can only deal with JAX, NumPy, Python or list type objects
# so to get around this, we are going to pass the data structure which
# stores the model's representation of the potential, i.e, theta
# print('jitchebwaveobject(thetatrue) =', fourwaveobject(thetatrue))
print('jitchebwaveobject(thetatrue.theta) =', jitfourwaveobject(thetatrue.theta))


jitfourwaveobjectjax = jax.jit(jax.grad(fourwaveobject))

# precompile chebwavegradsjax with Chebyshev representation
# of the true potential
print('nl.norm(jitchebwavegradsjax(vtruecheb)) =', nl.norm(jitfourwaveobjectjax(thetatrue.theta)))


###############################################################
# adjoint method for computing gradient
###############################################################

# function for computing gradients using adjoint method
def fourwavegradsadj(theta):
    #################################################
    # theta is a vector containing the concatenation
    # of the real and imaginary parts of vmat
    # its size should be
    # 2 * numtoepelms - 1 = 4 * numfour + 1
    #################################################

    # # to use theta we need to first recombine the real
    # # and imaginary parts into a vector of complex values
    # vtoephatR = theta[:numtoepelms]
    # vtoephatI = jnp.concatenate((jnp.array([0.0]), theta[numtoepelms:]))
    # vtoephat = vtoephatR + 1j * vtoephatI
    #
    # # construct vmathat from complex toeplitz vector
    # vhatmat = jnp.concatenate([jnp.flipud(jnp.conj(vtoephat)), vtoephat[1:]])[toepindxmat]

    # construct vmathat using the model class method
    # .tovmat(), theta is what ever the model class
    # uses as the data structure to store the potential
    # any other arguments are what is required to define
    # the model
    vhatmat = model.thetatovmat(theta, *modelprms)

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # **************************************************
    # Construct Hamiltonian matrix
    hhatmat = kmat + vhatmat

    # eigen-decomposition of the Hamiltonian matrix
    spchat, stthat = jnl.eigh(hhatmat)

    # compute propagator matrices
    propahat = stthat @ jnp.diag(jnp.exp(-1j * spchat * dt)) @ stthat.conj().T
    proplam = jnp.transpose(jnp.conjugate(propahat))

    # forward propagation
    ahatmatvec = []
    lammatvec = []
    for r in range(a0vec.shape[0]):
        ####################################################
        # build ahatmat, i.e., forward propagate of the
        # system with theta starting from a given initial
        # state a0
        ####################################################

        # initialize thisahatmat with given a0 state
        thisahatmat = [a0vec[r].copy()]

        # forward propagate of the system with theta
        for i in range(numts):
            # propagate the system one time-step and store the result
            thisahatmat.append(propahat @ thisahatmat[-1])

        # store compute ahatmat
        ahatmatvec.append(jnp.array(thisahatmat))

        ####################################################
        # build lammat
        # \lambda_N = (\hat{a}_N - a_N)
        # \lambda_j = (\hat{a}_j - a_j) + [\nabla_a \phi_{\Delta t} (a_j; \theta)]^\dagger \lambda_{j+1}
        # [\nabla_a \phi_{\Delta t} (a_j; \theta)]^\dagger = [exp{-i H \Delta t}]^\dagger
        ####################################################

        # compute the error of ahatmatvec[r] WRT amattruevec[r]
        thisahatmaterr = ahatmatvec[r] - amattruevec[r]

        # initialize thislammat
        thislammat = [thisahatmaterr[-1]]

        # build lammat backwards then flip
        for i in range(2, numts + 2):
            thislammat.append(thisahatmaterr[-i] + proplam @ thislammat[-1])

        # flip thislammat, make into a jax array, and add to lammatvec
        lammatvec.append(jnp.flipud(jnp.array(thislammat)))

    # make lists into JAX array object
    ahatmatvec = jnp.array(ahatmatvec)
    lammatvec = jnp.array(lammatvec)
    # **************************************************


    ####################################################
    # the remainder of this function is for computing
    # the gradient of the exponential matrix
    # - The blocks of code surrounded by '# ***...'
    #   are common to the adjoint method when using
    #   the Fourier basis to discretize, regardless
    #   of the model
    ####################################################

    # **************************************************
    # the code enclosed by ' # ****' is the same regardless
    # of what model you use
    # - Given the diagonalization H = U D U^\dagger
    # - The final gradient \nabla_\theta \phi(a;\theta)
    #   is Q = U M U^\dagger, where M = A (*) mask
    #   and A = U^\dagger [\nabla_\theta H = \nabla_\theta model of vhatmat or v(x)] U
    # **************************************************
    offdiagmask = jnp.ones((numtoepelms, numtoepelms)) - jnp.eye(numtoepelms)
    expspec = jnp.exp(-1j * dt * spchat)
    e1, e2 = jnp.meshgrid(expspec, expspec)
    s1, s2 = jnp.meshgrid(spchat, spchat)
    denom = offdiagmask * (-1j * dt) * (s1 - s2) + jnp.eye(numtoepelms)
    mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)

    # get the gradient of the model, function expects
    # model.thetatograd(theta, *args), where args is
    # what ever elements the model needs to be fully defined
    modelgrad = model.thetatograd(theta, *modelprms)

    # this line computes U^\dagger \nabla_\theta H(\theta) U
    prederivamat = jnp.einsum('ij,jkm,kl->ilm', stthat.conj().T, modelgrad, stthat)

    # this line computes
    # M_{i l} = A_{i l} (exp(D_{i i}) or (exp(D_{i i}) - exp(D_{l l}))/(D_{i i} - D_{l l}))
    derivamat = prederivamat * jnp.expand_dims(mask, 2)

    # this line computes Q = U M U^\dagger
    alldmat = -1j * dt * jnp.einsum('ij,jkm,kl->mil', stthat, derivamat, stthat.conj().T)

    # compute all entries of the gradient at once
    gradients = jnp.real(jnp.einsum('bij,ajk,bik->a', jnp.conj(lammatvec[:, 1:]), alldmat, ahatmatvec[:, :-1]))
    # **************************************************

    return gradients


# jist adjgrads
jitfourwavegradsadj = jax.jit(fourwavegradsadj)

# precompile chebwavegradsjax with Chebyshev representation
# of the true potential
print('nl.norm(jitchebwavegradsadj(vtruecheb)) =', nl.norm(jitfourwavegradsadj(thetatrue.theta)))