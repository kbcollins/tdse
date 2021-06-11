import time
import numpy
import scipy.integrate


###############################################################################
def norm_gauss(x, mean=0, std_dev=1):
    return numpy.exp(-0.5*((x-mean)/std_dev)**2) / (std_dev*numpy.sqrt(2*numpy.pi))


###############################################################################
def hermite_functions(x,
                      coefficient_array):
    
    n = len(coefficient_array)
    
    # the bigger n the more higher frequency terms are in the
    # resulting function
    
    two_power_n_array = 2**numpy.arange(n)
    
    factorial_array = numpy.array([numpy.arange(1, i).prod() for i in range(1, n+1)])
    
    norm_array = 1 / numpy.sqrt(numpy.sqrt(numpy.pi)*two_power_n_array * factorial_array)
    
    return numpy.exp(-0.5 * x**2) * numpy.polynomial.Hermite(norm_array * coefficient_array)(x)


###############################################################################
def random_hf(x,
              num_basis_span, # int 1-15
              num_basis_pool, # int 1-num_basis_span
              rtn_complex, # bool
              gauss_smooth=False, # bool
              gauss_mean=0,
              gauss_std_dev=1,
              gauss_hf_gain=0.25,
              seed=None):
    
    # initialize random bit generator
    rand_gen = numpy.random.default_rng(seed=seed)

    # make array with same number elements as index of largest nonzero basis
    if rtn_complex:
        coeff_array = numpy.zeros(num_basis_span, dtype=complex)
    else:
        coeff_array = numpy.zeros(num_basis_span)
    
    # select num_basis_span of randomly numbers from range [0, 1)
    real_coeff = rand_gen.random(num_basis_pool)
    
    # transform coefficent range to [-1, 1)
    # I should update this so it just randomly assigns neg signs
    # because this method includes 0 which I think slightly over
    # reps zero in the dist
    real_coeff = 2 * real_coeff - 1 
    
    # select which bases will have non-zero coefficients
    if num_basis_pool != num_basis_span:
        rand_basis_select = rand_gen.integers(low=0,
                                              high=num_basis_span,
                                              size=num_basis_pool
                                             )
    
    if rtn_complex:
        # generate two arrays of random numbers num_basis_span long
        imag_coeff = rand_gen.random(num_basis_pool)

        # make symmetric about zero
        imag_coeff = 2 * imag_coeff - 1
        
        all_rnd_coeff = real_coeff + 1j*imag_coeff
    else:
        all_rnd_coeff = real_coeff

    
    if num_basis_pool != num_basis_span:
        coeff_array[rand_basis_select] = all_rnd_coeff
    else:
        coeff_array = all_rnd_coeff
        
    
    if gauss_smooth:
        # make Gaussian
        gaussian = -norm_gauss(x, mean=gauss_mean, std_dev=gauss_std_dev)
    
        # make linear combo of HF
        hf = hermite_functions(x, coeff_array)

        # return randomly generated Hermite function w/ hf noise
        return gaussian + gauss_hf_gain*hf
    else:
        # return linear combo of Hermite functions
        return hermite_functions(x, coeff_array)


###############################################################################
# Random initial wavefunctions
# wrapper for generating some number of wavefunctions randomly
def random_psi0(num_psi0, space_steps):
    
    psi0_array = []
    
    for i in range(num_psi0):
        # Check if the generated wavefunctino is properly normalized
        # should it be?
        psi0_array.append(random_hf(space_steps,
                                    num_basis_span=7,# the lower the number, the smoother the wave function
                                    num_basis_pool=3, # int 1-15
                                    rtn_complex=True,
                                    seed=None))

    return numpy.array(psi0_array, dtype=complex)


###############################################################################
# Random potential
# have this be a wraper that selects which function you use and returns it as
# a function ready to use in the propagate procedure
#
# Idea: randomly select values for the mean, std_dev, and gain
# for all generated potentials
def random_potentials(num_potentials, space_steps):
    
    potential_function_array = []

    for i in range(num_potentials):
        potential_function_array.append(random_hf(space_steps,
                                                  num_basis_span=5, # smaller smoother
                                                  num_basis_pool=3,
                                                  rtn_complex=False,
                                                  gauss_smooth=True,
                                                  gauss_mean=0,
                                                  gauss_std_dev=1,
                                                  gauss_hf_gain=0.25,
                                                  seed=None))

    return numpy.array(potential_function_array)



###############################################################################
#
#                    PROPAGATION OF WAVE FUNCTION
#
###############################################################################
def sudo_spec_propagate(psi0, # array
                        potential, # array or function of discretized potential V(t) 
                        time_steps, # array
                        space_steps, # array
                        potential_time_dependent=False): 
    
    
    #######################################################################  
    num_space_steps = space_steps.shape[0]
    space_resolution = space_steps[1] - space_steps[0]
    
    # discretized (2*pi*j*k)^2
    k_res = 1/(num_space_steps * space_resolution)
    k_steps = numpy.arange(-num_space_steps//2, num_space_steps//2) * k_res
    k_terms = numpy.fft.ifftshift(2 * numpy.pi * 1j * k_steps)
    
    def d2z(psi):
        psi_hat = numpy.fft.fft(psi)

        # second derivative in Fourier space
        psi_hat_prime = k_terms**2 * psi_hat

        # transform back to signal space
        psi_xx = numpy.fft.ifft(psi_hat_prime)
        
        return psi_xx
    
    
    #######################################################################    
    num_time_steps = time_steps.shape[0]
    time_series_potential = numpy.empty((num_time_steps, num_space_steps))
        
    if potential_time_dependent:
        def sudo_spec_ode(time, psi):
            return (-0.5 * d2z(psi) + potential(time) * psi)/1j
        
        # propagate potential
        for i in range(num_time_steps):
            temp_time = time_steps[i]
            time_series_potential[i] = potential(temp_time)
        
    else:
        def sudo_spec_ode(time, psi):
            return (-0.5 * d2z(psi) + potential * psi)/1j
        
        # propagate potential
        time_series_potential = numpy.broadcast_to(potential, (num_time_steps, num_space_steps))
    
    
    #######################################################################
    # solve ODE using scipy
    time_series_psi = scipy.integrate.solve_ivp(sudo_spec_ode,
                                                t_span = (time_steps[0], time_steps[-1]),
                                                y0 = psi0,
                                                t_eval = time_steps)#,
                                                #rtol = 1e-10,
                                                #atol = 1e-10)

    # transpose solution so order of indices is (time, space)
    time_series_psi = time_series_psi.y.T
    
    
    #######################################################################
    return numpy.array([time_series_psi, time_series_potential])