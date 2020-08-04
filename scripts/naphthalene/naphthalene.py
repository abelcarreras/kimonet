from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step, system_test_info
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling, forster_coupling_extended
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes import GoldenRule, DecayRate, DirectRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.fileio import store_trajectory_list, load_trajectory_list
from kimonet.analysis import plot_polar_plot
from kimonet import calculate_kmc, calculate_kmc_parallel
from kimonet.system.state import State

import numpy as np
#np.random.seed(1)  # for testing


transfer_scheme = [GoldenRule(initial_states=('s1', 'gs'), final_states=('gs', 's1'),
                              electronic_coupling_function=forster_coupling_extended,
                              description='Forster',
                              arguments={'longitude': 2, 'n_divisions': 100}),

                   #DirectRate(initial=('s1', 's1'), final=('s1', 's1'),
                   #           rate_constant_function=forster_coupling,
                   #           description='ForsterX'),
                   ]

decay_scheme = [DecayRate(initial_states='s1', final_states='gs',
                          decay_rate_function=einstein_radiative_decay,
                          description='singlet_radiative_decay')
                ]

# reorganization energies of the states (eV)
reorganization_energies = {('s1', 'gs'): 0.08,
                           ('gs', 's1'): 0.08}

from scipy.interpolate import interp1d
from scipy.integrate import simps

data_abs = np.loadtxt('naphthalene_abs.txt').T
data_em = np.loadtxt('naphthalene_em.txt').T

n_abs = simps(y=data_abs[1], x=data_abs[0])
n_em = simps(y=data_em[1], x=data_em[0])

f_abs = interp1d(data_abs[0], data_abs[1]/n_abs, fill_value=0, bounds_error=False)
f_em = interp1d(data_em[0], data_em[1]/n_em, fill_value=0, bounds_error=False)

molecule = Molecule(states=[State(label='gs', energy=0, multiplicity=1),  # energies in eV
                            State(label='s1', energy=4.37, multiplicity=1)],  # energies in eV
                    # vibrations=MarcusModel(reorganization_energies),
                    vibrations=EmpiricalModel({('gs', 's1'): f_abs,
                                               ('s1', 'gs'): f_em}),
                    transition_moment={('s1', 'gs'): [0.9226746648, -1.72419493e-02, 4.36234688e-05],
                                      #('s1', 'gs'): np.array([2.26746648e-01, -1.72419493e-02, 4.36234688e-05]),
                                       ('s2', 'gs'): [2.0, 0.0, 0.0]},  # transition dipole moment of the molecule (Debye)
                    decays=decay_scheme,
                    vdw_radius=1.7
                    )


#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'refractive_index': 1}            # refractive index of the material (adimensional)

#######################################################################################################################

num_trajectories = 50                          # number of trajectories that will be simulated
max_steps = 10000                              # maximum number of steps for trajectory allowed

system_1 = regular_system(conditions=conditions,
                          molecule=molecule,
                          lattice={'size': [4, 4, 4], 'parameters': [3.0, 3.0, 3.0]},  # Angstroms
                          orientation=[0.8492, 1.0803, 0.4389])  # (Rx, Ry, Rz) if None then random orientation


system_2 = crystal_system(conditions=conditions,
                          molecules=[molecule, molecule],
                          scaled_site_coordinates=[[0, 0, 0],
                                                   [0.5, 0.5, 0.0]
                                                   ],
                          unitcell=[[6.32367864, 0.0000000, -4.35427391],
                                    [0.00000000, 5.7210000,  0.00000000],
                                    [0.00000000, 0.0000000,  8.39500000]],
                          dimensions=[2, 2, 2],
                          orientations=[[-2.4212, -1.8061,  1.9804],  # if element is None then random, if list then oriented
                                        [ 2.4212, -1.8061, -1.9804],
                                        None])


system = system_2  # choose 2
system.transfer_scheme = transfer_scheme
system.cutoff_radius = 8
system.add_excitation_index('s1', 0)
#system.add_excitation_center('s1')

# Donor: 558 / Acceptor: 447
# Donor: 558 / Acceptor: 540
from kimonet.core.processes.couplings import forster_coupling
#print(system.molecules[558].state)
#ec = forster_coupling(system.molecules[558], system.molecules[540], conditions, system.supercell)
#print(ec)


#print('-----')
# Donor: 558 / Acceptor: 1448 <- normal
#ec = forster_coupling(system.molecules[558], system.molecules[1448], conditions, system.supercell)
#print('n', ec)


system_test_info(system)

#visualize_system(system, dipole='s1')
system.add_excitation_index('s1', 0)

# do the kinetic Monte Carlo simulation
trajectories = calculate_kmc_parallel(system,
                                      processors=2,
                                      num_trajectories=50,    # number of trajectories that will be simulated
                                      max_steps=10,         # maximum number of steps for trajectory allowed
                                      silent=False)

# diffusion properties
analysis = TrajectoryAnalysis(trajectories)


for state in analysis.get_states():
    print('\nState: {}\n--------------------------------'.format(state))
    print('diffusion coefficient: {} angs^2/ns'.format(analysis.diffusion_coefficient(state)))
    print('lifetime: {} ns'.format(analysis.lifetime(state)))
    print('diffusion length: {} angs'.format(analysis.diffusion_length(state)))
    print('diffusion tensor (angs^2/ns)')
    print(analysis.diffusion_coeff_tensor(state))
    print(analysis.diffusion_coeff_tensor(state, unit_cell=system.supercell))

    print('diffusion length tensor (angs)')
    print(analysis.diffusion_length_square_tensor(state))
    print(analysis.diffusion_length_square_tensor(state, unit_cell=system.supercell))

    plot_polar_plot(analysis.diffusion_coeff_tensor(state),
                    title='Diffusion', plane=[0, 1])

    plot_polar_plot(analysis.diffusion_coeff_tensor(state, unit_cell=system.supercell),
                    title='Diffusion', crystal_labels=True, plane=[0, 1])

    plot_polar_plot(analysis.diffusion_coeff_tensor(state, unit_cell=system.supercell),
                    title='Diffusion', crystal_labels=True, plane=[0, 2])

    plot_polar_plot(analysis.diffusion_length_square_tensor(state, unit_cell=system.supercell),
                    title='Length square', crystal_labels=True, plane=[0, 1])

    plot_polar_plot(analysis.diffusion_length_square_tensor(state, unit_cell=system.supercell),
                    title='Length square', crystal_labels=True, plane=[0, 1])

    plot_polar_plot(analysis.diffusion_length_square_tensor(state, unit_cell=system.supercell),
                    title='Length square', crystal_labels=True, plane=[0, 2])

exit()
store_trajectory_list(trajectories, 'test.h5')

trajectory_list = load_trajectory_list('test.h5')


print('--------**************----------')
analysis = TrajectoryAnalysis(trajectory_list)


for state in analysis.get_states():
    print('\nState: {}\n--------------------------------'.format(state))
    print('diffusion coefficient: {} angs^2/ns'.format(analysis.diffusion_coefficient(state)))
    print('lifetime: {} ns'.format(analysis.lifetime(state)))
    print('diffusion length: {} angs'.format(analysis.diffusion_length(state)))
    print('diffusion tensor (angs^2/ns)')
    print(analysis.diffusion_coeff_tensor(state))
    print('diffusion length tensor (angs)')
    print(analysis.diffusion_length_square_tensor(state))

exit()


#plt.figure(1)
#analysis.plot_excitations('s1')
#plt.show()
#analysis.plot_excitations('s2')
#analysis.plot_excitations()

analysis.plot_2d('s1').show()
#plt.figure()
#plt = analysis.plot_2d('s2')
analysis.plot_distances('s1').show()
#plt.figure()
#analysis.plot_distances('s2')
#analysis.plot_histogram('s1').savefig('test.png')
analysis.plot_histogram('s1').show()

#plt.figure()
#analysis.plot_histogram('s2')

