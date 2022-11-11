# Example of the calculation of the emission/absorption spectrum for methyl peroxy radical
# using Levich Jortner Model
from kimonet.system.vibrations import MarcusModel, LevichJortnerModel
from kimonet.system.state import State
from kimonet.core.processes.transitions import Transition
import matplotlib.pyplot as plt
import numpy as np



excitation_energy = 1.1278 # eV
reorganization = 0.1456 # eV
temperature = 50

# frequencies
freq_i = [140.79, 491.95, 918.28, 1135.28, 1160.57, 1225.62, 1456.34, 1492.36, 1501.75, 3074.37, 3168.4, 3182.64] # cm-1
freq_f = [251.97, 374.87, 929.19, 1045.94, 1171.51, 1180.31, 1463.87, 1485.01, 1524.9, 3048.78, 3122.53, 3175.95] # cm-1

# Huang Rhys factors
s_i = [0.0, 0.08545292342599448, 0.0019509010632587217, 3.187932781959664e-35, 0.17270848729824412, 0.35738637524607225,
       0.003142694268121429, 1.0476585878561509e-35, 0.003333073460643237, 0.0001898943302040377, 0.0, 0.0006556995989193498]
s_f = [0.0, 0.07258444675793045, 0.08793170887680193, 0.2838510623430307, 3.2896687455020486e-35, 0.07301678176745915,
       0.0002811971489664261, 0.0, 0.0014814731195096128, 0.0003244944182861039, 0.0, 0.00039346347817862163]

# states list
gs = State(label='gs', energy=0, nm_frequencies=freq_i)
s1 = State(label='s1', energy=excitation_energy, nm_frequencies=freq_f)

# transitions
transitions = [Transition(s1, gs,
                          reorganization_energy=reorganization/2,  # eV
                          huang_rhys=s_i,
                          symmetric=False,
                          ),
               Transition(gs, s1,
                          reorganization_energy=reorganization/2,  # eV
                          huang_rhys=s_f,
                          symmetric=False,
                          ),
               ]

# load model
lj = LevichJortnerModel(transitions=transitions,
                        temperature=temperature
                        )

# get spectrum functions
spectrum_em = lj.get_vib_spectrum(s1, gs)
spectrum_abs = lj.get_vib_spectrum(gs, s1)

# plot data
energies = np.linspace(excitation_energy-0.8, excitation_energy+0.8, 500)
plt.plot(energies, spectrum_abs(energies), '-', label='Absorption', color='C3')
plt.plot(energies, spectrum_em(energies), '-', label='Emission', color='C4')
plt.xlabel('Energy [eV]')
plt.ylabel('Intensity [eV-1]')
plt.yticks([], [])
plt.legend()
plt.show()
