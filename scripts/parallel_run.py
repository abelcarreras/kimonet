# This works only in python 3
from kimonet.system.generators import ordered_system, disordered_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet import update_system
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_singlet_decay
from kimonet.core.processes import Transfer, Decay
import kimonet.core.processes as processes
import concurrent.futures as futures


processes.transfer_scheme = {Transfer(initial=('s1', 'gs'), final=('gs', 's1'), description='forster'): forster_coupling,
                             # Transfer(initial=('s1', 'gs'), final=('gs', 's2'), description='test'): forster_coupling,
                             # Transfer(initial=('s2', 'gs'), final=('gs', 's1'), description='test2'): forster_coupling,
                             # Transfer(initial=('s2', 'gs'), final=('gs', 's2'), description='test3'): forster_coupling
                             }

decay_scheme = {Decay(initial='s1', final='gs', description='singlet_radiative_decay'): einstein_singlet_decay,
                # Decay(initial='s1', final='s2', description='singlet_radiative_decay'): singlet_decay,
                # Decay(initial='s2', final='gs', description='singlet_radiative_decay'): singlet_decay,
                }

# excitation energies of the electronic states (eV)
state_energies = {'gs': 0,
                  's1': 1}

# reorganization energies of the states (eV)
reorganization_energies = {'gs': 0,
                           's1': 0.2}

molecule = Molecule(state_energies=state_energies,
                    reorganization_energies=reorganization_energies,
                    transition_moment=[2.0, 0],  # transition dipole moment of the molecule (Debye)
                    decays=decay_scheme
                    )

#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'temperature': 273.15,            # temperature of the system (K)
              'refractive_index': 1,            # refractive index of the material (adimensional)
              'cutoff_radius': 3.1}             # maximum interaction distance (Angstroms)

#######################################################################################################################

num_trajectories = 500                          # number of trajectories that will be simulated
max_steps = 100000                              # maximum number of steps for trajectory allowed

system = ordered_system(conditions=conditions,
                        molecule=molecule,
                        lattice={'size': [3, 3], 'parameters': [3.0, 3.0]},  # Angstroms
                        orientation=[0, 0, 0])

# visualize_system(system)


def run_trajectory(system, index):

    system = system.copy()
    system.add_excitation_center('s1')

    trajectory = Trajectory(system)
    for i in range(max_steps):

        change_step, step_time = update_system(system)

        if system.is_finished:
            break

        trajectory.add(change_step, step_time)

    print('trajectory {} done!'.format(index))
    return trajectory


# executor = futures.ThreadPoolExecutor(max_workers=4)
executor = futures.ProcessPoolExecutor()

futures_list = []
for i in range(num_trajectories):
    futures_list.append(executor.submit(run_trajectory, system, i))

trajectories = []
for f in futures.as_completed(futures_list):
    trajectories.append(f.result())


analysis = TrajectoryAnalysis(trajectories)

print('diffusion coefficient (average): {} angs^2/ns'.format(analysis.diffusion_coefficient()))
print('lifetime: {} ns'.format(analysis.lifetime()))
print('diffusion length: {} angs'.format(analysis.diffusion_length()))

print('diffusion tensor')
print(analysis.diffusion_coeff_tensor('s1'))
print('diffusion length tensor')
print(analysis.diffusion_length_tensor('s1'))
# print(np.sqrt(analysis.diffusion_coeff_tensor()*analysis.lifetime()*2))

plt = analysis.plot_2d()
plt.figure()
analysis.plot_distances()
