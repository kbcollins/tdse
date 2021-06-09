import time
import numpy
import scipy.integrate
from HermiteFunctionsV5 import random_hf
from HermiteFunctionsV5 import random_complex_hf


def training_data_generator(num_samples = 100,
                            num_potentials = 1,
                            potential_amp = 1,
                            test_set_split = 0.2, # percent
                            test_set_rnd_seed = None,
                            num_time_steps = 500,
                            initial_time = 0, # seconds
                            final_time = 5, # seconds
                            num_space_steps = 1024,
                            left_space_bound = -25.6, # space units
                            right_space_bound = 25.6, # space units
                           ):
    
    print(f'Building training data, {num_samples} samples:')
    
    # store start time
    start_time = time.time() 

    # initialize mid_time
    mid_time = start_time
    
    # initialize current_time
    current_time = start_time
    
    # define time grid
    time_steps, time_res = numpy.linspace(initial_time,
                                      final_time,
                                      num_time_steps,
                                      retstep=True
                                     )
    
    # define space grid
    space_steps, space_res = numpy.linspace(left_space_bound,
                                        right_space_bound,
                                        num_space_steps,
                                        retstep=True
                                       )
    
    
    #######################################################
    # Random initial wavefunctions
    
    print('>>> Generating random initial wavefunctions')
    mid_time = time.time()
    
    # generate random initial wavefunctions
    psi0_tensor = numpy.array(
        [random_complex_hf(space_steps, smoothing=1.5)
         for _ in range(num_samples)]
    )

    current_time = time.time()
    print(f'>>> Success {current_time - mid_time}')
    mid_time = current_time
    
    
    #######################################################
    # Random potential functions
    
    print('>>> Generating random potentials')
    mid_time = time.time()
    
    # potential function evaluated at space_steps
    potential_tensor = numpy.array(
        [random_hf(space_steps,
                   num_basis=3,
                   variance=1,
                   hf_amp=1
                  ) for _ in range(num_potentials)]
    )
    
    print(potential_tensor.shape)
    
    current_time = time.time()
    print(f'>>> Success {current_time - mid_time}')
    mid_time = current_time
    
    
    #######################################################
    # Pseudo-spectral ODE
    
    # discretized (2*pi*j*k)^2
    k_res = 1/(num_space_steps * space_res)
    k_steps = numpy.arange(-num_space_steps//2, num_space_steps//2) * k_res
    k_terms = numpy.fft.ifftshift(2 * numpy.pi * 1j * k_steps)
    
    # pseudo spectral ODE
    def sudo_spec_ode(t, psi, index=0):
        psi_hat = numpy.fft.fft(psi)

        # second derivative in Fourier space
        psi_hat_prime = k_terms**2 * psi_hat

        # transform back to signal space
        psi_xx = numpy.fft.ifft(psi_hat_prime) 
        rhs = (-0.5 * psi_xx + potential_tensor[index] * psi)/1j
        return rhs
    
    # initialize sol_tensor
    sol_tensor = numpy.array([numpy.empty((num_samples,
                              num_time_steps,
                              num_space_steps),
                            dtype=complex) for i in range(num_potentials)])

    print('>>> Solving pseudo-spectral ODE...')
    ode_solve_start_time = time.time()
    mid_time = ode_solve_start_time
    
    for p in range(num_potentials):
        # solve pseudo spectral ODE using scipy
        for i in range(num_samples):

            # Give progress message every 10 samples solved
            if i%10 == 0: #and i != 0:
                current_time = time.time()
                print(f'    {i} of {num_samples},',
                      f'Batch time: {current_time - mid_time}')
                # update value in mid_time with current time
                mid_time = current_time

            # solve sample using scipy routine
            sol = scipy.integrate.solve_ivp(
                sudo_spec_ode,
                args=(p,),
                t_span = (initial_time, final_time),
                y0 = psi0_tensor[i],
                t_eval = time_steps,
                #vectorized = True,
                rtol = 1e-10,
                atol = 1e-10)

            # transpose solution so order of indices is (time, space)
            sol_tensor[p,i] = sol.y.T

            # tranpose solution so order is (time, space)
            # and normalize trajectories
            # sol_tensor[p,i] = sol.y.T / numpy.max(sol.y.T)
    
    # total time to solve all ODEs
    current_time = time.time()
    print(f'>>> Success {current_time - ode_solve_start_time}')
    
    
    #######################################################
    # prepare return 
    
    total_train_in = []
    total_train_out = []
    total_test_in = []
    total_test_out = []
    
    for p in range(num_potentials):
        num_test_samples = int(num_samples*test_set_split)
        
        test_set_index_array = numpy.random.default_rng(test_set_rnd_seed).integers(
            0, num_samples, num_test_samples)

        train_input = numpy.delete(sol_tensor[p], test_set_index_array, axis=0)
        total_train_in.append(train_input)
        
        train_output = potential_tensor[p].copy()
        total_train_out.append(numpy.broadcast_to(train_output, train_input.shape))
        
        test_input = sol_tensor[p,test_set_index_array]
        total_test_in.append(test_input)
        
        test_output = potential_tensor[p].copy()
        total_test_out.append(test_output)
        

    total_train_in = numpy.array(total_train_in)
    total_train_out = numpy.array(total_train_out)
    total_test_in = numpy.array(total_test_in)
    total_test_out = numpy.array(total_test_out)
    
    
    
    # total time
    print(f'Total elapse time: {time.time() - start_time}')
    
    return ({'input': total_train_in, 'output': total_train_out},
            {'input': total_test_in, 'output': total_test_out},
            time_steps,
            space_steps)