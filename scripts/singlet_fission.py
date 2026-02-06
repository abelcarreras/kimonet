# Tetracene exemaple
from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis, plot_polar_plot
from kimonet import system_test_info
from kimonet.system.molecule import Molecule
from kimonet.system.state import State
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate, SimpleRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.core.processes.transitions import Transition
from kimonet import calculate_kmc, calculate_kmc_parallel, calculate_kmc_parallel_py2
from kimonet.system.state import ground_state as gs
from kimonet.utils import old_distance_between_molecules, distance_vector_periodic
from kimonet.fileio import store_trajectory_list, load_trajectory_list
import numpy as np

# states list
s1 = State(label='s1', energy=2.9705, multiplicity=1, size=1)
tt = State(label='tt', energy=2.0, multiplicity=1, size=2, connected_distance=8)
t1 = State(label='t1', energy=1.5, multiplicity=3, size=1)


# transition moments
transition_moment = {Transition(s1, gs): [0.1, 0.0]}


def direction_transfer(initial, final, rate_constant=1):
    """
    Allows transfer along a single direction

    :param initial:
    :param final:
    :param rate_constant:
    :return:
    """

    r = initial[0].get_center().get_coordinates() - initial[1].get_center().get_coordinates()
    dot = np.dot(r, initial[0].get_center().get_orientation_vector())

    #if np.abs(dot) > 0.1:
    #    rate_constant = 0

    return rate_constant


def electronic_coupling_direction(initial, final, couplings=None):
    """
    Allows transfer along a single direction

    :param initial: initial states list
    :param final: final states list
    :param couplings: coupling list
    :return: rate constant
    """

    r_vector = initial[1].get_center().get_coordinates() - initial[0].get_center().get_coordinates()
    cell_incr = initial[0].cell_state - final[0].cell_state

    r = distance_vector_periodic(r_vector, initial[0].supercell, cell_incr)

    norm = np.linalg.norm(r)
    dot_a = np.abs(np.dot(r, [1, 0]))/np.linalg.norm([1, 0])/norm
    dot_ab_1 = np.abs(np.dot(r, [1, 1]))/np.linalg.norm([1, 1])/norm
    dot_ab_2 = np.abs(np.dot(r, [1, -1]))/np.linalg.norm([1, -1])/norm

    dot_b = np.abs(np.dot(r, [0, 1]))/np.linalg.norm([0, 1])/norm
    ichosen = int(np.argmax([dot_a, dot_ab_1, dot_ab_2, dot_b]))

    return couplings[ichosen]  # eV


# Electronic couplings in eV for the closest neighbor molecule in the indicated direction
singlet_couplings = [12.61e-3,  # a
                     41.85e-3,  # ab
                     41.85e-3,  # ab
                     27.51e-3]  # b

triplet_couplings = [0.0e-3,  # a
                     7.2e-3,  # ab
                     7.2e-3,  # ab
                     1.2e-3]  # b
# Vibrations
vibrational_model = MarcusModel(transitions=[Transition(gs, s1, reorganization_energy=0.07),
                                             Transition(s1, gs, reorganization_energy=0.07),
                                             Transition(gs, t1, reorganization_energy=0.07),
                                             Transition(t1, gs, reorganization_energy=0.07)],
                                temperature=300)

#################################################################################

# 2D model (plane a-b) , not diffusion in C
molecule = Molecule()

system = crystal_system(molecules=[molecule, molecule],  # molecule to use as reference
                        scaled_site_coordinates=[[0.0, 0.0],
                                                 [0.5, 0.5]],
                        unitcell=[[7.3347, 0.0000],
                                  [-0.2242, 6.0167]],
                        dimensions=[4, 4],  # supercell size
                        orientations=[[0.0, 0.0, np.pi/8],
                                      [0.0, 0.0, -np.pi/8]])  # if element is None then random, if list then Rx Ry Rz

system.cutoff_radius = 8.1  # Angstroms

                         # Transport
system.process_scheme = [GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                                    electronic_coupling_function=electronic_coupling_direction,
                                    arguments={'couplings': singlet_couplings},
                                    vibrations=vibrational_model,
                                    description='singlet transport'),
                         GoldenRule(initial_states=(t1, gs), final_states=(gs, t1),
                                    electronic_coupling_function=electronic_coupling_direction,
                                    arguments={'couplings': triplet_couplings},
                                    vibrations=vibrational_model,
                                    description='triplet transport'),
                         # Transitions
                         SimpleRate(initial_states=(s1, gs), final_states=(tt,),
                                    rate_constant=8.3,  # ns^-1
                                    description='singlet fission'),
                         SimpleRate(initial_states=(tt,), final_states=(gs, s1),
                                    rate_constant=1.0,  # ns^-1
                                    description='triplet fusion'),
                         SimpleRate(initial_states=(tt,), final_states=(t1, t1),
                                    rate_constant=2.0,  # ns^-1
                                    description='triplet dissociation'),
                         # Decays
                         SimpleRate(initial_states=(s1,), final_states=(gs,),
                                    rate_constant=8e-2,  # ns^-1
                                    description='Singlet decay '),
                         SimpleRate(initial_states=(t1,), final_states=(gs,),
                                    rate_constant=1.6e-5,  # ns^-1
                                    description='Triplet decay'),
                         SimpleRate(initial_states=(t1, t1), final_states=(s1, gs),
                                    rate_constant=1.8e-6,  # ns^-1
                                    description='triplet-triplet annihilation'),
                         #SimpleRate(initial_states=(s1, s1), final_states=(gs, gs),
                         #           rate_constant=1,
                         #           description='singlet-singlet annihilation'),
                         #SimpleRate(initial_states=(t1, s1), final_states=(gs, gs),
                         #           rate_constant=100,
                         #           description='singlet-triplet annihilation'),
                         ]

np.random.seed(0)

system.add_excitation_random(s1, 2)
system_test_info(system)
visualize_system(system)

# exit()
trajectories = calculate_kmc(system,
                             num_trajectories=50,    # number of trajectories that will be simulated
                             max_steps=1000,         # maximum number of steps for trajectory allowed
                             silent=False)


store_trajectory_list(trajectories, 'singlet_fission.h5')
#trajectories = load_trajectory_list('singlet_fission.h5')

analysis = TrajectoryAnalysis(trajectories)

for s in ['s1', 't1']:
    print('STATE: ', s)
    print('diffusion coefficient (average): {} angs^2/ns'.format(analysis.diffusion_coefficient(s)))
    print('lifetime: {} ns'.format(analysis.lifetime(s)))
    print('diffusion length: {} angs'.format(analysis.diffusion_length(s)))

    print('diffusion tensor')
    print(analysis.diffusion_coeff_tensor(s))
    print(analysis.diffusion_coeff_tensor(s, unit_cell=system.supercell))
    print('diffusion length tensor')
    print(analysis.diffusion_length_square_tensor(s))
    print(analysis.diffusion_length_square_tensor(s, unit_cell=system.supercell))

    # print(np.sqrt(analysis.diffusion_coeff_tensor()*analysis.lifetime()*2))

    plt = analysis.plot_2d(state=s)
    plt.figure()
    analysis.plot_distances(state=s)
    plt.show()

    plot_polar_plot(analysis.diffusion_coeff_tensor(s))
    plot_polar_plot(analysis.diffusion_coeff_tensor(s, unit_cell=system.supercell))


for s in ['s1', 't1', 'tt']:
    analysis.plot_exciton_density(state=s)
plt = analysis.plot_exciton_density()
plt.show()