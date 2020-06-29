import numpy as np
from kimonet.utils.units import SPEED_OF_LIGHT, HBAR_PLANCK
from kimonet import _ground_state_


# Decay functions
def einstein_radiative_decay(molecule, g1=1, g2=1):
    """
    Einstein radiative decay

    :param molecule:
    :param g1: degeneracy of target state
    :param g2: degeneracy of origin state

    :return: decay rate constant
    """
    deexcitation_energy = molecule.state.energy - molecule.get_state_energy(_ground_state_)

    mu2 = np.dot(molecule.get_transition_moment(), molecule.get_transition_moment())  # transition moment norm.
    alpha = 1.0 / 137.036
    return float(g1)/g2 * alpha * 4 * deexcitation_energy ** 3 * mu2 / (3 * SPEED_OF_LIGHT ** 2 * HBAR_PLANCK ** 3)


def triplet_triplet_annihilation(molecule):
    f = 1
    ct = 1
    ptta = 50
    lifetime = 10
    a0 = 0.529177249
    t = ct*a0**-3
    return 1/(f*lifetime*t)*(1/ptta-1)**-1
