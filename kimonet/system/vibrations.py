import numpy as np
import warnings
from kimonet.utils.units import BOLTZMANN_CONSTANT, HBAR_PLANCK
from scipy.integrate import quad
import math


class Vibrations:

    def __init__(self,
                 temperature=300,
                 frequencies=None,
                 reorganization_energies=None,  # eV
                 external_reorganization_energies=None,  # eV
                 state_energies=None
                 ):

        self.temperature = temperature
        self.frequencies = frequencies
        self.reorganization_energies = reorganization_energies
        self.external_reorganization_energies = external_reorganization_energies
        self.state_energies = state_energies

        # symetrize external reorganization energies
        if self.external_reorganization_energies is not None:
            for key in list(external_reorganization_energies):
                external_reorganization_energies[key[::-1]] = external_reorganization_energies[key]

    def set_state_energies(self, state_energies):
        self.state_energies = state_energies

    def get_marcus_vib_spectrum(self, transition):
        warnings.warn('Using Marcus method')

        elec_trans_ene = self.state_energies[transition[1]] - self.state_energies[transition[0]]

        temp = self.temperature  # temperature (K)
        reorg_ene = np.sum(self.reorganization_energies[transition])

        sign = np.sign(elec_trans_ene)

        # print('T', temp, molecule.reorganization_energies[transition], elec_trans_ene, -sign)

        def vib_spectrum(e):
            return 1.0 / (np.sqrt(4.0 * np.pi * BOLTZMANN_CONSTANT * temp * reorg_ene)) * \
                   np.exp(-(elec_trans_ene - e * sign + reorg_ene) ** 2 / (4 * BOLTZMANN_CONSTANT * temp * reorg_ene))

        return vib_spectrum

    def levich_jortner_vib_spectrum(self, transition):

        elec_trans_ene = self.state_energies[transition[1]] - self.state_energies[transition[0]]

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
