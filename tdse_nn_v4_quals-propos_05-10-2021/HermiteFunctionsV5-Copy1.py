import numpy


def hermite_functions(x, lin_combo_coeff, smoothing=0):
    """
    This generates a linear combination whose bases are the Hermite
    functions and whose coefficients are given by lin_combo_coeff. It then
    applies the resulting function to an input x. An optional smoothing
    parameter is provided which causes the higher order bases to have less
    effect, i.e., it scales the k-th coefficient like (1/k)^smoothing.
    
    lin_combo_coeff: array of coefficients. 15 bases seems to be about as high
    as is possible
    
    x: Where f(x) = linear combo of hermite function bases, x is what the
    resulting function is appled to.
    
    smoothing: a real number.
    """
    n = len(lin_combo_coeff)
    
    smoothing_array = 1 / (numpy.arange(1, n+1) ** smoothing)
    
    two_squared_array = 2**numpy.arange(n)
    
    factorial_array = numpy.array(
        [numpy.arange(1, i).prod() for i in range(1, n+1)])
    
    norm_array = 1 / numpy.sqrt(numpy.sqrt(numpy.pi)
                                     * two_squared_array * factorial_array)
    
    return numpy.exp(-0.5 * x**2) * numpy.polynomial.Hermite(
        smoothing_array * norm_array * lin_combo_coeff)(x)


###############################################################################
def random_hf(x,
              num_basis,
              max_num_basis=15,
              variance = 5, # variance of Gaussian
              hf_amp = 0.25,
              seed=None):
    
    # initialize random bit generator
    rand_gen = numpy.random.default_rng(seed=seed)
    
    # select which bases will have non-zero coefficients
    rand_basis_select = rand_gen.integers(low=0,
                                          high=max_num_basis,
                                          size=num_basis
                                         )
    
    # make an array whose largest index is equal to the largest nonzero basis
    coeff_array = numpy.zeros(rand_basis_select.max()+1)
    
    # randomly select the coefficents the range is [0, 1)
    rand_coeff = rand_gen.random(num_basis)
    
    # transform coefficent range to [-1, 1)
    rand_coeff = 2 * rand_coeff - 1
    
    # load the randomly generated coefficents into coeff_array for the selected
    # non-zero coefficents
    coeff_array[rand_basis_select] = rand_coeff
    
    # get Hermite function signal
    hf = hermite_functions(x, coeff_array, smoothing=0) # ADDED SMOOTHING 05/10/2021
    
    # initialize Gaussian signal
    gaussian = numpy.exp(-0.5 / variance**2 * x**2)
    gaussian /= numpy.max(numpy.abs(gaussian)) # make so abs(max) of Gaussian = 1 ADDED 05/10/2021
    #gaussian /= gaussian.max() # make abs(max) of Gaussian 1 COMMENTS OUT 05/10/2021
    gaussian *= -1 # invert Gaussian so it is a bowl
    
    # return randomly generated Hermite function
    return gaussian + hf_amp*hf # add hf to Gaussian like a noise


###############################################################################
def random_complex_hf(x, num_basis=15, smoothing=0, seed=None):
    # generate two arrays of random numbers num_basis long
    r, c = numpy.random.default_rng(seed=seed).random((2, num_basis))
    
    # make random numbers symmetric about zero
    r = 2 * r - 1
    c = 2 * c - 1
    
    rand_array = r + 1j*c
    return hermite_functions(x, rand_array, smoothing)
    