import numpy as np
from kimonet.utils.units import SPEED_OF_LIGHT, HBAR_PLANCK
from kimonet.core.processes.transitions import Transition
from kimonet.system.state import ground_state as _GS_
from kimonet.utils.units import ATOMIC_TO_ANGS_EL


# Decay functions
def einstein_radiative_decay(initial, final, g1=1, g2=1, transitions=None, transition_moment=None):
    """
    Einstein radiative decay

    :param molecule:
    :param g1: degeneracy of target state
    :param g2: degeneracy of origin state

    :return: decay rate constant
    """
    deexcitation_energy = initial[0].energy - final[0].energy

    mu = np.array(transitions[transitions.index(Transition(initial[0], final[0]))].tdm) * ATOMIC_TO_ANGS_EL

    # mu = np.array(transition_moment[Transition(initial[0], final[0])]) * ATOMIC_TO_ANGS_EL
    mu2 = np.dot(mu, mu)  # transition moment square norm.
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
