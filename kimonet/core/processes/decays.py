import numpy as np
from kimonet.utils.units import SPEED_OF_LIGHT, HBAR_PLANCK


# Decay functions
def einstein_singlet_decay(molecule):
    desexcitation_energy = molecule.state_energies[molecule.state] - molecule.state_energies['gs']
    mu2 = np.dot(molecule.transition_moment, molecule.transition_moment)  # transition moment norm.
    alpha = 1.0 / 137
    return alpha * 4 * desexcitation_energy ** 3 * mu2 / (3 * SPEED_OF_LIGHT ** 2 * HBAR_PLANCK ** 3)

