from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import do_simulation_step, system_test_info
from kimonet.core.processes.couplings import forster_coupling, dexter_coupling, forster_coupling_extended
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.fileio import store_trajectory_list, load_trajectory_list
from kimonet.analysis import plot_polar_plot
from kimonet import calculate_kmc, calculate_kmc_parallel
from kimonet.system.state import State
from kimonet.system.state import ground_state as gs
from kimonet.core.processes.transitions import Transition
import numpy as np

#np.random.seed(1)  # for testing

s1 = State(label='s1', energy=4.37, multiplicity=1)
s2 = State(label='s2', energy=4.37, multiplicity=1)


# transition moments
transitions = [Transition(s1, gs,
                          tdm=[0.9226746648, -1.72419493e-02, 4.36234688e-05],  # a.u.
                          reorganization_energy=0.08),
               Transition(s2, gs,
                          tdm=[0.9226746648, -1.72419493e-02, 4.36234688e-05],  # a.u.
                          reorganization_energy=0.08),
               ]

from scipy.interpolate import interp1d
from scipy.integrate import simpson as simps

data_abs = np.loadtxt('naphthalene_abs.txt').T
data_em = np.loadtxt('naphthalene_em.txt').T

n_abs = simps(y=data_abs[1], x=data_abs[0])
n_em = simps(y=data_em[1], x=data_em[0])

f_abs = interp1d(data_abs[0], data_abs[1]/n_abs, fill_value=0, bounds_error=False)
f_em = interp1d(data_em[0], data_em[1]/n_em, fill_value=0, bounds_error=False)


process_scheme = [GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                              electronic_coupling_function=forster_coupling_extended,
                              description='Forster',
                              arguments={'longitude': 2, 'n_divisions': 100, 'transitions': transitions, 'ref_index': 1.0},
                              vibrations=EmpiricalModel({Transition(gs, s1, symmetric=False): f_abs,  # eV
                                                         Transition(s1, gs, symmetric=False): f_em})  # eV

                             ),

                  DecayRate(initial_state=s1, final_state=gs,
                            decay_rate_function=einstein_radiative_decay,
                            arguments={'transitions': transitions},
                            description='singlet_radiative_decay'),
                  ]

molecule = Molecule(name='naphtahlene', site_energy=2, vdw_radius=1.7)

system_1 = regular_system(molecule=molecule,
                          lattice={'size': [2, 2, 2], 'parameters': [3.0, 3.0, 3.0]},  # Angstroms
                          orientation=[0.8492, 1.0803, 0.4389])  # (Rx, Ry, Rz) if None then random orientation

system_2 = crystal_system(molecules=[molecule, molecule],
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
system.process_scheme = process_scheme
system.cutoff_radius = 8
system.add_excitation_index(s1, 0)

# system_test_info(system)
#visualize_system(system, length=2)

# do the kinetic Monte Carlo simulation
trajectories = calculate_kmc_parallel(system,
                                      processors=4,
                                      num_trajectories=10,    # number of trajectories that will be simulated
                                      max_steps=5000,         # maximum number of steps for trajectory allowed
                                      silent=False)

# diffusion properties
analysis = TrajectoryAnalysis(trajectories)

analysis.plot_distances('s1').show()
print('diffusion length S1: {} angs'.format(analysis.diffusion_length('s1')))
print('lifetime: {} ns'.format(analysis.lifetime('s1')))

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

# store trajectory
store_trajectory_list(trajectories, 'test.h5')


load_file = False
if load_file:
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

