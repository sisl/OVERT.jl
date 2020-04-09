# Control Environments

This repo includes:
1. n-link pendulum implementation with point masses m at the end of each link. Links are massless. Viscous friction with a coefficient c is assumed. The links are actuated. This is a 2n dimensional dynamical system. [th1, th2, ..., thn, dth1, dth2, ..., dthn]. Parameters include:
	-  `n_pend`: number of links
        -  `m`: mass of point masses (default value = 0.5 kg)
        -  `L`: length of each link (default value = 0.5 m)
        -  `c`: viscous friction coefficient (default value = 0.1 N.s)
        -  `g`: gravitational acceleration (default value = 9.8 m/s^2)
        -  `max_action`: maximum torque on each link (default value = 1000 N.m)
        -  `x_0`: initial position (default value = `[0.]*2*n_pend`)
        -  `integration_method`: integration scheme, options are "1st" which is a first order euler update and "2nd"
                                 which is crank-nicolson update. 
        -  `dt`: timestep (default value = 0.001 s)
    
The single, double and triple pendulums are explicitly implemented. For higher-n's, one needs to add a function that include the dynamics of the system.
2. airplane:
