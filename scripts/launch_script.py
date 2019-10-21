from kimonet.system import ordered_system, disordered_system
from kimonet.core import update_system
from kimonet.analysis import Trajectory, visualize_system, TrajectoryAnalysis
from kimonet.molecules import Molecule
import json
import warnings
import numpy as np
np.random.seed(0)


"""
Generic molecule initialization
Possible states: 
    'gs': ground state 
    's1': first singlet state 
All energies must be given in eV. By default initialized at gs.
"""

# excitation energies of the electronic states (eV)
state_energies = {'gs': 0,
                  's1': 1}

# reorganization energies of the states (eV)
reorganization_energies = {'gs': 0,
                           's1': 0.2}

molecule = Molecule(state_energies=state_energies,
                    reorganization_energies=reorganization_energies,
                    transition_moment=[2.0, 0]  # transition dipole moment of the molecule (Debye)
                    )

#######################################################################################################################

# physical conditions of the system (as a dictionary)
conditions = {'temperature': 273.15,            # temperature of the system (K)
              'refractive_index': 1,            # refractive index of the material (adimensional)
              'cutoff_radius': 3.1}             # maximum interaction distance (Angstroms)

#######################################################################################################################

num_trajectories = 50                          # number of trajectories that will be simulated
max_steps = 100000                              # maximum number of steps for trajectory allowed

system = ordered_system(conditions=conditions,
                        molecule=molecule,
                        lattice={'size': [3, 3], 'parameters': [3.0, 3.0]},  # Angstroms
                        orientation=[0, 0, 0])

# visualize_system(system)

trajectories = []
for j in range(num_trajectories):

    system.add_excitation_center('s1')
    # system.add_excitation_index('s1', 0)

    # visualize_system(system)

    print('iteration: ', j)

    trajectory = Trajectory(system)
    for i in range(max_steps):

        change_step, step_time = update_system(system)

        if system.is_finished:
            break

        trajectory.add(change_step, step_time)

        # visualize_system(system)

        if i == max_steps-1:
            warnings.warn('Maximum number of steps reached!!')

    system.reset()

    trajectories.append(trajectory)

z = 2

# diffusion properties
d_tensor = np.average([traj.get_diffusion_tensor(0) for traj in trajectories], axis=0)
print('diffusion tensor')
print(d_tensor)

d_coefficient = np.average([traj.get_diffusion(0) for traj in trajectories])
lifetime = np.average([traj.get_lifetime(0) for traj in trajectories])
print('diffusion coefficient (average): {}angs^2/ns'.format(d_coefficient))
print('lifetime: {} ns'.format(lifetime))

length2 = np.average([traj.get_diffusion_length_square(0) for traj in trajectories])
print('diffusion length: ', np.sqrt(length2), np.sqrt(2 * z * d_coefficient * lifetime), 'angs')
print('diffusion(2)', length2 / (lifetime * 2 * z))

# print('--------------TEST2--------------')
analysis = TrajectoryAnalysis(trajectories)

print(analysis)

print('diffusion coefficient (average): {}angs^2/ns'.format(analysis.diffusion_coefficient()))
print('lifetime: {} ns'.format(analysis.lifetime()))
print('diffusion length: {} angs'.format(analysis.diffusion_length()))

print('diffusion tensor')
print(analysis.diffusion_coeff_tensor())
print('diffusion length tensor')
print(analysis.diffusion_length_tensor())
# print(np.sqrt(analysis.diffusion_coeff_tensor()*analysis.lifetime()*2))


plt = analysis.plot_2d()
plt.figure()
analysis.plot_distances()

plt.show()

exit()


#######################################################################################################################

output_file_name = '2_excitons_simulation_1d.json'      # name of the output file where the trajectories will be saved
                                                        # .json format

#######################################################################################################################


# We collect all the outputs in a dictionary system_information and write it in the output_file

system_information = {'conditions': system['conditions'],
                      'lattice': system['lattice'],
                      'orientation': 'parallel',
                      'type': system['type'],
                      'excitons': {'s1': ['centre']},
                      'state_energies': state_energies,
                      'reorganization_energies': reorganization_energies,
                      'transition_moment': list(molecule.transition_moment)}

output = {'system_information': system_information,
          'trajectories': [traj for traj in trajectories],
          'max_steps': max_steps}

with open(output_file_name, 'w') as write_file:
    json.dump(output, write_file)


################# Parallel test (py3 only) ##################

import concurrent.futures as futures


def function_test(a, b, test=0):
    import time
    time.sleep(abs(i-5))
    return a * b + test


executor = futures.ThreadPoolExecutor()

fut = []
for i in range(10):
    fut.append(executor.submit(function_test, i, i))

for f in futures.as_completed(fut):
    print(f.result())

for f in fut:
    print(f.result(), f.done(), f.running())
