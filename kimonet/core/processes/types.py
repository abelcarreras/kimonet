#from kimonet.core.processes.fcwd import general_fcwd
from kimonet.utils.units import HBAR_PLANCK
import numpy as np
from kimonet.system.vibrations import NoVibration
from scipy.integrate import quad
from copy import deepcopy
from kimonet.system.state import ground_state as _GS_


overlap_data = {}


def ordered_states(state_list):
    """
    get GS states behind and other states at front
    :param state_list: List of states behind and
    :return: ordered List
    """
    ordered_list = []
    for state in state_list:
        if state.label == _GS_.label:
            ordered_list.append(state)
        else:
            ordered_list.insert(0, state)

    return tuple(ordered_list)


class BaseProcess:
    def __init__(self,
                 initial_states,
                 final_states,
                 description='',
                 arguments=None
                 ):

        self.initial = ordered_states(initial_states)
        self._final = None
        self.final_test = deepcopy(final_states)

        self.description = description
        self.cell_states = {}
        self.arguments = arguments if arguments is not None else {}
        self._supercell = None

        # Check input coherence
        total_size_initial = np.sum([state.size for state in initial_states])
        total_size_final = np.sum([state.size for state in final_states])

        assert total_size_initial == total_size_final

    #    def __str__(self):
#        return 'donor/acceptor : {} {}\n'.format(self.donor.state, self.acceptor.state) \
#               + 'initial : {} {}\n'.format(self.initial[0], self.initial[1]) \
#               + 'final : {} {}\n'.format(self.final[0], self.final[1])

    @property
    def final(self):
        if self._final is None:
            self._final = deepcopy(self.final_test)
            for state in self._final:
                for mol in state.get_molecules():
                    mol.cell_state = self.cell_states[mol]
                    mol.set_state(state)

        return self._final


    @property
    def supercell(self):
        if self._supercell is None:
            raise Exception('No supercell')
        return self._supercell

    @supercell.setter
    def supercell(self, cell):
        self._supercell = cell
        for state in self.final_test:
            state.supercell = cell

    def reset_cell_states(self):
        self.cell_states.clear()
        for state in self.final_test:
            for mol in state.get_molecules():
                self.cell_states[mol] = np.zeros(mol.get_dim())

    def get_molecules(self):

        molecules_list = []
        for state in self.initial:
            molecules_list += state.get_molecules()

        return molecules_list

class GoldenRule(BaseProcess):
    def __init__(self,
                 initial_states,
                 final_states,
                 electronic_coupling_function,
                 description='',
                 arguments=None,
                 vibrations=NoVibration(),
                 ):

        self._coupling_function = electronic_coupling_function
        self._vibrations = vibrations
        BaseProcess.__init__(self, initial_states, final_states, description, arguments)

    @property
    def vibrations(self):
        return self._vibrations

    def get_fcwd(self):
        transition_donor = (self.initial[0], self.final[0])
        transition_acceptor = (self.initial[1], self.final[1])

        donor_vib_dos = self.vibrations.get_vib_spectrum(*transition_donor)  # (transition_donor)
        acceptor_vib_dos = self.vibrations.get_vib_spectrum(*transition_acceptor)  # (transition_acceptor)

        # print(donor_vib_dos)
        info = str(hash(donor_vib_dos) + hash(acceptor_vib_dos))

        # the memory is used if the overlap has been already computed
        if info in overlap_data:
            return overlap_data[info]

        def overlap(x):
            return donor_vib_dos(x) * acceptor_vib_dos(x)

        overlap_data[info] = quad(overlap, 0, np.inf, epsabs=1e-5, limit=1000)[0]

        return overlap_data[info]

    def get_electronic_coupling(self, conditions):
        # conditions will be deprecated
        return self._coupling_function(self.initial, self.final, conditions, self.supercell, **self.arguments)

    def get_rate_constant(self, conditions, *args):
        e_coupling = self.get_electronic_coupling(conditions)
        # spectral_overlap = general_fcwd(self.donor, self.acceptor, self, conditions)

        spectral_overlap = self.get_fcwd()

        return 2 * np.pi / HBAR_PLANCK * e_coupling ** 2 * spectral_overlap  # Fermi's Golden Rule


class DirectRate(BaseProcess):
    def __init__(self,
                 initial_states,
                 final_states,
                 rate_constant_function,
                 description='',
                 arguments=None
                 ):

        self.rate_function = rate_constant_function
        BaseProcess.__init__(self, initial_states, final_states, description, arguments)

    def get_rate_constant(self, conditions, *args):
        return self.rate_function(self.initial, self.final, conditions, self.supercell)


class DecayRate(BaseProcess):
    def __init__(self,
                 initial_states,
                 final_states,
                 decay_rate_function,
                 description='',
                 arguments=None
                 ):

        BaseProcess.__init__(self, [initial_states], [final_states], description, arguments)
        self.rate_function = decay_rate_function

    def get_rate_constant(self, *args):
        return self.rate_function(self.initial, self.final, **self.arguments)
