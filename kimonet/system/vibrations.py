import numpy as np
import warnings
from kimonet.utils.units import BOLTZMANN_CONSTANT, HBAR_PLANCK
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from kimonet.core.processes.transitions import Transition


class MarcusModel:

    def __init__(self,
                 transitions=None,
                 temperature=300,  # Kelvin
                 drift_term=0  # aN -> eV / angs [to be implemented]
                 ):


        self.temperature = temperature
        self._transitions = transitions


    def __hash__(self):
        #return hash((str(self.reorganization_energies)))
        return hash((str(self._transitions)))

    def get_vib_spectrum(self, target_state, origin_state):

        # elec_trans_ene = self.state_energies[transition[1]] - self.state_energies[transition[0]]
        elec_trans_ene = target_state.energy - origin_state.energy
        transition = Transition(target_state, origin_state, symmetric=False)

        temp = self.temperature  # temperature (K)

        if transition in self._transitions:
            reorg_ene = self._transitions[self._transitions.index(transition)].reorganization_energy
        #if transition in self.reorganization_energies:
        #    reorg_ene = np.sum(self.reorganization_energies[transition])
        else:
            raise Exception('Reorganization energy for transition {} not defined'.format(transition))

        sign = np.sign(elec_trans_ene)

        # print('T', temp, molecule.reorganization_energies[transition], elec_trans_ene, -sign)

        def vib_spectrum(e):
            return 1.0 / (np.sqrt(4.0 * np.pi * BOLTZMANN_CONSTANT * temp * reorg_ene)) * \
                   np.exp(-(elec_trans_ene - e * sign + reorg_ene) ** 2 / (4 * BOLTZMANN_CONSTANT * temp * reorg_ene))

        return vib_spectrum


class LevichJortnerModel:

    def __init__(self,
                 frequencies=None,
                 reorganization_energies=None,  # eV
                 external_reorganization_energies=None,  # eV
                 temperature=300,
                 # state_energies=None,
                 ):

        self.frequencies = frequencies
        self.reorganization_energies = reorganization_energies
        self.external_reorganization_energies = external_reorganization_energies
        self.state_energies = None
        self.temperature = temperature

        # symmetrize external reorganization energies
        """
        if self.external_reorganization_energies is not None:
            for key in list(external_reorganization_energies):
                external_reorganization_energies[key[::-1]] = external_reorganization_energies[key]
        """

    def __hash__(self):
        return hash((str(self.state_energies),
                     str(self.frequencies),
                     str(self.external_reorganization_energies),
                     str(self.reorganization_energies)))

    def set_state_energies(self, state_energies):
        self.state_energies = state_energies

    def get_vib_spectrum(self, target_state, origin_state):

        elec_trans_ene = target_state.energy - origin_state.energy
        transition = Transition(target_state, origin_state, symmetric=False)

        #elec_trans_ene = self.state_energies[transition[1]] - self.state_energies[transition[0]]

        temp = self.temperature  # temperature (K)
        ext_reorg_ene = self.external_reorganization_energies[transition]
        reorg_ene = np.array(self.reorganization_energies[transition])

        sign = np.sign(elec_trans_ene)

        angl_freqs = np.array(self.frequencies[transition]) * 29.9792558 * 2*np.pi  # cm-1 -> ns-1 , angular frequency

        freq_classic_limit = BOLTZMANN_CONSTANT * temp / HBAR_PLANCK  # ns^-1,  angular frequency

        indices_cl = np.where(angl_freqs < freq_classic_limit)[0]
        indices_qm = np.where(angl_freqs >= freq_classic_limit)[0]

        l_cl = np.sum(reorg_ene[indices_cl]) + ext_reorg_ene

        l_qm = reorg_ene[indices_qm]

        ang_freq_qm = angl_freqs[indices_qm]

        s = np.true_divide(l_qm, ang_freq_qm) / HBAR_PLANCK

        s_eff = np.sum(s)
        w_eff = np.sum(np.multiply(s, ang_freq_qm))/s_eff  # angular frequency

        # print('l_qm', np.sum(l_qm))
        # print('s_eff', s_eff)
        # print('w_eff', w_eff)

        def vib_spectrum(e):
            e = np.array(e, dtype=float)
            fcwd_term = np.zeros_like(e)
            for m in range(10):
                fcwd_term += s_eff**m / np.math.factorial(m) * np.exp(-s_eff) * np.exp(
                    -(elec_trans_ene - e * sign + l_cl + m * HBAR_PLANCK * w_eff)**2 / (4 * BOLTZMANN_CONSTANT * temp * l_cl))

            return 1.0 / (np.sqrt(4 * np.pi * BOLTZMANN_CONSTANT * temp * l_cl)) * fcwd_term

        return vib_spectrum


class EmpiricalModel:

    def __init__(self,
                 empirical_function=None,  # eV
                 ):

        self.empirical_function = empirical_function
        self.state_energies = None

    def __hash__(self):
        return hash((str(self.state_energies),
                     str(self.empirical_function)))

    def set_state_energies(self, state_energies):
        self.state_energies = state_energies

    def get_vib_spectrum(self, target_state, origin_state):
        transition = Transition(target_state, origin_state, symmetric=False)

        # Temperature is not actually used. This is to keep common interface
        return self.empirical_function[transition]


class GaussianModel:

    def __init__(self,
                 deviations=None,
                 reorganization_energies=None,  # eV
                 # state_energies=None,
                 ):

        self.reorganization_energies = reorganization_energies
        self.state_energies = None
        self.deviations = deviations

        # symmetrize external reorganization energies
        """
        if self.external_reorganization_energies is not None:
            for key in list(external_reorganization_energies):
                external_reorganization_energies[key[::-1]] = external_reorganization_energies[key]
        """

    def __hash__(self):
        return hash((str(self.state_energies),
                     str(self.deviations),
                     str(self.reorganization_energies)))

    def get_vib_spectrum(self, target_state, origin_state):

        """
        :param donor: energy diference between states
        :param acceptor: deviation in energy units
        :return: Franck-Condon-weighted density of states in gaussian aproximation
        """

        elec_trans_ene = target_state.energy - origin_state.energy
        transition = Transition(target_state, origin_state, symmetric=False)

        #elec_trans_ene = self.state_energies[transition[1]] - self.state_energies[transition[0]]
        reorg_ene = np.sum(self.reorganization_energies[transition])

        deviation = self.deviations[transition]     # atomic units

        sign = np.sign(elec_trans_ene)
        def vib_spectrum(e):
            return np.exp(-(elec_trans_ene - e * sign + reorg_ene) ** 22 / (2 * deviation) ** 2) / (2 * np.sqrt(np.pi) * deviation)

        return vib_spectrum


class NoVibration:

    def __init__(self,
                 ):

        self.state_energies = None

    def __hash__(self):
        return hash((str(self.state_energies)))

    def set_state_energies(self, state_energies):
        self.state_energies = state_energies

    def get_vib_spectrum(self, target_state, origin_state):

        elec_trans_ene = target_state.energy - origin_state.energy

        return elec_trans_ene

        #transition = Transition(target_state, origin_state, symmetric=False)

        # elec_trans_ene = self.state_energies[transition[1]] - self.state_energies[transition[0]]

        #def vib_spectrum(e):
        #    if elec_trans_ene == e:
        #        return 1
        #    else:
        #        return 0
        #
        #return vib_spectrum


class SimpleBoltzmann:

    def __init__(self,
                 temperature=300,  # Kelvin
                 ):

        self.temperature = temperature
        self.state_energies = None

    def __hash__(self):
        return hash((str(self.state_energies)))

    def set_state_energies(self, state_energies):
        self.state_energies = state_energies

    def get_vib_spectrum(self, target_state, origin_state):

        elec_trans_ene = target_state.energy - origin_state.energy

        return np.exp(-elec_trans_ene/(BOLTZMANN_CONSTANT * self.temperature))


# Analysis functions
def get_normalized_spectrum(x, y, in_nm=False, interpolation='quadratic'):

    warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)

    _EV_TO_NM_ = 1239.841

    if in_nm:
        x = _EV_TO_NM_ / np.array(x)

    # from scipy.integrate import quad
    # def integrand(x, donor, acceptor):
    #    return f_d(x) * f_a(x)
    # fcwd = quad(integrand, 1.e-8, np.inf, args=(0, 0),  epsabs=1e-20)[0]

    # Normalize spectra
    normalization_constant = abs(integrate.simps(y, x))
    print('normalize', normalization_constant)

    y /= normalization_constant

    f_x = interpolate.interp1d(x, y, fill_value=0, bounds_error=False, kind=interpolation)

    return f_x


def print_data(f_donor, f_acceptor):

    def overlap(x):
        return f_donor(x) * f_acceptor(x)

    x_new = np.linspace(1.5, 4, 1000)

    fcwd_test = integrate.simps(f_donor(x_new) * f_acceptor(x_new), x_new)

    test_donor = integrate.quad(f_donor, 0, np.inf,  epsabs=1e-20)[0]
    test_acceptor = integrate.quad(f_acceptor, 0, np.inf,  epsabs=1e-20)[0]
    test_overlap = integrate.quad(overlap, 0, np.inf,  epsabs=1e-20)[0]

    print('Integral, Donor: {} Acceptor: {} Overlap: {}'.format(test_donor, test_acceptor, test_overlap))
    print('FCWD : {}'.format(fcwd_test))

    center_em = np.average(x_new, weights=f_donor(x_new))
    center_abs = np.average(x_new, weights=f_acceptor(x_new))

    print('Center abs: {} eV'.format(center_abs))
    print('Center em: {} eV'.format(center_em))

    print('Reorganization energy: {} eV'.format(np.abs(center_abs - center_em)))
