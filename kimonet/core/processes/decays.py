import numpy as np
from kimonet.utils.units import SPEED_OF_LIGHT, HBAR_PLANCK
from kimonet import _ground_state_


# Decay functions
def einstein_singlet_decay(molecule):
    desexcitation_energy = molecule.state_energies[molecule.state] - molecule.state_energies[_ground_state_]

    mu2 = np.dot(molecule.get_transition_moment(), molecule.get_transition_moment())  # transition moment norm.
    alpha = 1.0 / 137
    return alpha * 4 * desexcitation_energy ** 3 * mu2 / (3 * SPEED_OF_LIGHT ** 2 * HBAR_PLANCK ** 3)


def triplet_triplet_annihilation(molecule):
    f = 1
    ct = 1
    ptta = 50
    lifetime = 10
    a0 = 0.529177249
    t = ct*a0**-3
    return 1/(f*lifetime*t)*(1/ptta-1)**-1
