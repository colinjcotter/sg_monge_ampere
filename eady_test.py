import matplotlib
matplotlib.use('Agg')

import numpy as np
from periodic_densities import Periodic_density_in_x, sample_rectangle

import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from eady_initial import initialise_points, eady_OT, forward_euler_sg, heun_sg
from eady_simulation_scripts import frontogenesis_timestep, validity_analysis_results

#set conditions for simulation
add_data = True   # save point and weight values
Heun = True       # use Heun's method for time integration
days = 25         # number of days 
N = 40            # grid points
tstepsize = 1800. # stepsize

print(frontogenesis_timestep(N, days, tstepsize, Heun))

print(validity_analysis_results(N, days, tstepsize, Heun))
