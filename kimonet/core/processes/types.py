from kimonet.core.processes.fcwd import general_fcwd
from kimonet.utils.units import HBAR_PLANCK
import numpy as np


class BaseProcess:
    def __init__(self,
                 initial,
                 final,
                 description='',
                 arguments=None
                 ):

        self.initial = initial
        self.final = final
        self.description = description
        self.arguments = arguments if arguments is not None else {}
        self._donor = None
        self._acceptor = None

    @property
    def donor(self):
        if self._donor is None:
            raise Exception('No donor set')
        return self._donor

    @donor.setter
    def donor(self, molecule):
        self._donor = molecule

    @property
    def acceptor(self):
        if self._acceptor is None:
            raise Exception('No acceptor set')
        return self._acceptor

    @acceptor.setter
    def acceptor(self, molecule):
        self._acceptor = molecule


class GoldenRule(BaseProcess):
    def __init__(self,
                 initial,
                 final,
                 electronic_coupling_function,
                 description='',
                 arguments=None
                 ):

        self._coupling_function = electronic_coupling_function
        BaseProcess.__init__(self, initial, final, description, arguments)

    def get_electronic_coupling(self, conditions, supercell, cell_incr):
        return self._coupling_function(self.donor, self.acceptor, conditions, supercell, cell_incr, **self.arguments)

    def get_rate_constant(self, conditions, supercell, cell_incr):
        e_coupling = self.get_electronic_coupling(conditions, supercell, cell_incr)
        spectral_overlap = general_fcwd(self.donor, self.acceptor, self, conditions)
        return 2 * np.pi / HBAR_PLANCK * e_coupling ** 2 * spectral_overlap  # Fermi's Golden Rule


class DirectRate(BaseProcess):
    def __init__(self,
                 initial,
                 final,
                 rate_constant_function,
                 description='',
                 arguments=None
                 ):

        self.rate_function = rate_constant_function
        BaseProcess.__init__(self, initial, final, description, arguments)

    def get_rate_constant(self, conditions, supercell, cell_incr):
        return self.rate_function(self.donor, self.acceptor, conditions, supercell, cell_incr)


class DecayRate(BaseProcess):
    def __init__(self,
                 initial,
                 final,
                 decay_rate_function,
                 description='',
                 arguments=None
                 ):

        BaseProcess.__init__(self, initial, final, description, arguments)
        self.rate_function = decay_rate_function

    def get_rate_constant(self, molecule):
        return self.rate_function(molecule, **self.arguments)
