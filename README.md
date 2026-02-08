[![Actions Status](https://github.com/abelcarreras/kimonet/actions/workflows/python-package.yml/badge.svg)](https://github.com/abelcarreras/kimonet/actions)
[![PyPI version](https://badge.fury.io/py/kimonet.svg)](https://badge.fury.io/py/kimonet)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abelcarreras/kimonet/)

Kimonet
=======
A kinetic Monte Carlo code to simulate exciton dynamics

Features
--------
- Multi-dimensional simulations (1D, 2D & 3D)
- High level of abstraction
- Easy to extend to custom processes

Requirements
------------
- Python 2.7.x/3.2+ 
- numpy
- scipy
- matplotlib
- networkx
- h5py
- pygraphviz (which requires graphviz-dev library) [optional]

1-D example
------------------------------------

```python
from kimonet.core.processes.types import DecayRate, DirectRate
from kimonet.system.molecule import Molecule
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system import System
from kimonet.system.state import State
from kimonet import system_test_info, calculate_kmc, calculate_kmc_parallel
from kimonet.system.state import ground_state as gs
import numpy as np


# custom transfer function
def transfer_rate(initial, final, custom_constant=1):

    # r_vector = initial[0].get_coordinates_absolute() - initial[1].get_coordinates_absolute()
    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()
    distance = np.linalg.norm(r_vector)

    return custom_constant/distance**2


# custom decay function
def decay_rate(initial, final):
    rates = {'TypeA': 1. / 100,  # ns-1
             'TypeB': 1. / 50,  # ns-1
             'TypeC': 1. / 25}  # ns-1
    return rates[initial[0].get_center().name]


# define states
s1 = State(label='s1', energy=1.0, multiplicity=1, size=1)
s2 = State(label='s2', energy=1.5, multiplicity=1)

# setup molecules in the simulation
molecule = Molecule()

molecule1 = molecule.copy()
molecule1.set_coordinates([0])
molecule1.name = 'TypeA'

molecule2 = molecule.copy()
molecule2.set_coordinates([1])
molecule2.name = 'TypeB'

molecule3 = molecule.copy()
molecule3.set_coordinates([2])
molecule3.name = 'TypeC'


# setup the system to be simulated
system = System(molecules=[molecule1, molecule2, molecule3],
                supercell=[[3]])

# set initial exciton positions
system.add_excitation_index(s1, 1)
system.add_excitation_index(s2, 2)


# set all processes in the simulation
system.process_scheme = [DirectRate(initial_states=(s1, gs), final_states=(gs, s1),
                                    rate_constant_function=transfer_rate,
                                    description='custom'),
                         DirectRate(initial_states=(s2, gs), final_states=(gs, s2),
                                    rate_constant_function=transfer_rate,
                                    description='custom'),
                         DecayRate(initial_state=s1, final_state=gs,  # TO SYSTEM
                                   decay_rate_function=decay_rate,
                                   description='custom decay rate'),
                         DecayRate(initial_state=s2, final_state=gs,  # TO SYSTEM
                                   decay_rate_function=decay_rate,
                                   description='custom decay rate')
                         ]
# set interaction cutoff radius in Angstrom
system.cutoff_radius = 10.0 

# system analysis functions
system_test_info(system)
visualize_system(system)

# run the kinetic Monte Carlo simulation
parallel_run = True
if parallel_run:
    trajectories = calculate_kmc_parallel(system,
                                          processors=6,
                                          num_trajectories=100,    # number of trajectories that will be simulated
                                          max_steps=100000,         # maximum number of steps for trajectory allowed
                                          silent=False)
else:
    trajectories = calculate_kmc(system,
                                 num_trajectories=10,    # number of trajectories that will be simulated
                                 max_steps=100000,         # maximum number of steps for trajectory allowed
                                 silent=False)


# Results analysis
analysis = TrajectoryAnalysis(trajectories)

print('diffusion coefficient: {:9.5e} Angs^2/ns'.format(analysis.diffusion_coefficient('s1')))
print('lifetime:              {:9.5e} ns'.format(analysis.lifetime('s1')))
print('diffusion length:      {:9.5e} Angs'.format(analysis.diffusion_length('s1')))
print('diffusion tensor (Angs^2/ns)')
print(analysis.diffusion_coeff_tensor('s1'))

print('diffusion length square tensor (Angs)')
print(analysis.diffusion_length_square_tensor('s1'))

analysis.plot_exciton_density('s1').show()
analysis.plot_distances('s1').show()
analysis.plot_histogram('s1', normalized=True, bins=20).show()
```

2-D example
------------------------------------

```python
from kimonet.system.generators import regular_system, crystal_system
from kimonet.analysis import visualize_system, TrajectoryAnalysis
from kimonet.system.molecule import Molecule
from kimonet.core.processes.couplings import forster_coupling
from kimonet.core.processes.decays import einstein_radiative_decay
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel, EmpiricalModel
from kimonet.fileio import store_trajectory_list, load_trajectory_list
from kimonet.analysis import plot_polar_plot
from kimonet.system.state import State
from kimonet.system.state import ground_state as gs
from kimonet.core.processes.transitions import Transition
from kimonet import calculate_kmc, calculate_kmc_parallel, calculate_kmc_parallel_py2
from kimonet import system_test_info
import numpy as np


# states list
s1 = State(label='s1', energy=20.0, multiplicity=1)

# transition moments
transitions = [Transition(s1, gs,
                          tdm=[0.1, 0.0],  # a.u.
                          reorganization_energy=0.08)]  # eV

# define molecules
molecule = Molecule()
molecule2 = Molecule(site_energy=2)

# set crystal structure
system = crystal_system(molecules=[molecule, molecule],  # molecule to use as reference
                        scaled_site_coordinates=[[0.0, 0.0],
                                                 [0.0, 0.5]],
                        unitcell=[[5.0, 1.0],
                                  [1.0, 5.0]],
                        dimensions=[2, 2],  # supercell dimensions
                        orientations=[[0.0, 0.0, np.pi/2],
                                      [0.0, 0.0, 0.0]]) 

# set initial exciton
system.add_excitation_index(s1, 0)
system.add_excitation_index(s1, 1)


# define rate constants function
def test_function(initial, final):
    """
    rate as a function of distance

    :param initial: initial states
    :param final: final states
    """

    r_vector = initial[0].get_coordinates_absolute() - initial[1].get_coordinates_absolute()
    return 1 /np.linalg.norm(r_vector)

# define processes
system.process_scheme = [GoldenRule(initial_states=(s1, gs), final_states=(gs, s1),
                                    electronic_coupling_function=forster_coupling,
                                    description='Forster coupling',
                                    arguments={'ref_index': 1, 'transitions': transitions},
                                    vibrations=MarcusModel(transitions=transitions) # eV
                                    ),
                         DecayRate(initial_state=s1, final_state=gs,
                                   decay_rate_function=einstein_radiative_decay,
                                   arguments={'transitions': transitions},
                                   description='custom decay rate')
                        ]

# interaction cutoff radius in Angstrom
system.cutoff_radius = 8  

# some system analyze functions
system_test_info(system)
visualize_system(system)

# run kinetic Monte Carlo simulation
trajectories = calculate_kmc(system,
                             num_trajectories=5,    # number of trajectories that will be simulated
                             max_steps=100,         # maximum number of steps for trajectory allowed
                             silent=False)

# specific trajectory plot
trajectories[0].plot_graph().show()
trajectories[0].plot_2d().show()

# resulting trajectories analysis
analysis = TrajectoryAnalysis(trajectories)

for state in analysis.get_states():
    print('\nState: {}\n--------------------------------'.format(state))
    print('diffusion coefficient: {:9.5e} Angs^2/ns'.format(analysis.diffusion_coefficient(state)))
    print('lifetime:              {:9.5e} ns'.format(analysis.lifetime(state)))
    print('diffusion length:      {:9.5e} Angs'.format(analysis.diffusion_length(state)))
    print('diffusion tensor (angs^2/ns)')
    print(analysis.diffusion_coeff_tensor(state))

    print('diffusion length tensor (Angs)')
    print(analysis.diffusion_length_square_tensor(state))

    plot_polar_plot(analysis.diffusion_coeff_tensor(state),
                    title='Diffusion', plane=[0, 1])

    plot_polar_plot(analysis.diffusion_length_square_tensor(state, unit_cell=[[5.0, 1.0],
                                                                              [1.0, 5.0]]),
                    title='Diffusion length square', crystal_labels=True, plane=[0, 1])


analysis.plot_exciton_density('s1').show()
analysis.plot_2d('s1').show()
analysis.plot_distances('s1').show()
analysis.plot_histogram('s1').show()
analysis.plot_histogram('s1').savefig('histogram_s1.png')
```


Contact info
------------
Abel Carreras  
abelcarreras83@gmail.com

Donostia International Physics Center (DIPC)  
Donostia-San Sebastian, Euskadi (Spain)