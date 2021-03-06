#from kimonet.core.processes.fcwd import general_fcwd
from kimonet.utils.units import HBAR_PLANCK
import numpy as np
from kimonet.system.vibrations import NoVibration
from scipy.integrate import quad
from copy import deepcopy
from kimonet.system.state import ground_state as _GS_
from kimonet.utils.combinations import combinations_group


overlap_data = {}


def ordered_states(state_list):
    """
    get GS states behind and other states at front
    :param state_list: List of states behind and
    :return: ordered List
    """
    labels = [s.label for s in state_list]
    indices = np.argsort(labels)
    state_list = np.array(state_list)[indices].tolist()

    ordered_list = []
    for state in state_list:
        if state.label == _GS_.label:
            ordered_list.append(state)
        else:
            ordered_list.insert(0, state)

    return tuple(ordered_list)


class BaseProcess(object):
    def __init__(self,
                 initial_states,
                 final_states,
                 description='',
                 arguments=None,
                 do_copy=True,
                 ):

        if do_copy:
            initial_states = tuple([s.copy() for s in initial_states])
            final_states = tuple([s.copy() for s in final_states])

        self._initial = ordered_states(initial_states)
        self._final_test = ordered_states(deepcopy(final_states))

        self._final = None

        self.description = description
        self.cell_states = {}
        self.arguments = arguments if arguments is not None else {}
        self._supercell = None
        self._transition_connect = None
        self._transport_connect = None
        self._is_symmetry = None

        # Check input coherence
        total_size_initial = np.sum([state.size for state in initial_states])
        total_size_final = np.sum([state.size for state in final_states])

        # Check initial & final states sizes match
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
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, state_list):
        self._transition_connect = None
        self._transport_connect = None
        self._initial = state_list

    @property
    def final_test(self):
        return self._final_test

    @final_test.setter
    def final_test(self, state_list):
        self._transition_connect = None
        self._transport_connect = None
        self._final_test = state_list

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

    def is_symmetry(self):
        if self._is_symmetry is None:
            label_list = [s.label for s in self._initial]
            self._is_symmetry = not len(np.unique(label_list)) == len(label_list)

        return self._is_symmetry

    def get_self_interaction_process(self):
        """
        By default not allow self interaction in this process
        :return:
        """
        return None

    def reset_cell_states(self):
        self.cell_states.clear()
        for state in self.final_test:
            for mol in state.get_molecules():
                self.cell_states[mol] = np.zeros(mol.get_dim())

    def get_molecules(self):

        molecules_list = []
        for state in self._initial:
            molecules_list += state.get_molecules()

        return molecules_list

    def get_transport_connections(self):
        """
        Get the connections between initial and final states
        that are considered to be transport (same state moving)
        :return:
        """
        if self._transport_connect is None:
            self._transport_connect = {}
            for istate in self._initial:
                for fstate in self._final_test:
                    if istate.label == fstate.label and istate.label != _GS_.label:
                        if istate in self._transport_connect:
                            self._transport_connect[istate].append(fstate)
                        else:
                            self._transport_connect[istate] = [fstate]

        return self._transport_connect

    def get_transition_connections(self):
        """
        Get the connections between initial and final states
        that are considered to be transitions (state converting to other)
        :return:
        """

        if self._transition_connect is None:
            self._transition_connect = {}
            inital_states = [s for s in self._initial if s.label != _GS_.label]
            final_states = [s for s in self._final_test if s.label != _GS_.label]
            if len(inital_states) == 1:
                self._transition_connect[inital_states[0]] = []
                for fstate in final_states:
                    self._transition_connect[inital_states[0]].append(fstate)
            elif len(final_states) == 1:
                for istate in inital_states:
                    self._transition_connect[istate] = [final_states[0]]
            else:
                for istate in self._initial:
                    self._transition_connect[istate] = []
                    for fstate in final_states:
                        self._transition_connect[istate].append(fstate)

        return self._transition_connect


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
        transition_donor = (self.initial[0], self.final[1])
        transition_acceptor = (self.initial[1], self.final[0])

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

    def get_electronic_coupling(self):
        return self._coupling_function(self.initial, self.final, **self.arguments)

    def get_rate_constant(self):
        e_coupling = self.get_electronic_coupling()
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

    def get_rate_constant(self):
        return self.rate_function(self.initial, self.final, **self.arguments)


class SimpleRate(BaseProcess):
    def __init__(self,
                 initial_states,
                 final_states,
                 rate_constant,
                 description='',
                 arguments=None
                 ):

        self._rate_constant = rate_constant
        BaseProcess.__init__(self, initial_states, final_states, description, arguments)

    def get_rate_constant(self):
        return self._rate_constant


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

    def get_rate_constant(self):
        return self.rate_function(self.initial, self.final, **self.arguments)
