from kimonet.utils.units import HBAR_PLANCK, BOLTZMANN_CONSTANT
import numpy as np
from kimonet.system.vibrations import NoVibration
from scipy.integrate import quad
from copy import deepcopy
from kimonet.system.state import ground_state as _GS_
import warnings


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
        self._cell_increment = None

        # Check initial & final states sizes match
        total_size_initial = np.sum([state.size for state in initial_states])
        total_size_final = np.sum([state.size for state in final_states])
        assert total_size_initial == total_size_final

        # Check initial & final states multiplicity coherence
        total_multiplicity_initial = np.sum([state.multiplicity-1 for state in initial_states])
        total_multiplicity_final = np.sum([state.multiplicity-1 for state in final_states])

        if (total_multiplicity_initial % 2 == 0) != (total_multiplicity_final % 2 == 0):
            warnings.warn('Multiplicity in process "{}" is not coherent'.format(self.description))

    #    def __str__(self):
#        return 'donor/acceptor : {} {}\n'.format(self.donor.state, self.acceptor.state) \
#               + 'initial : {} {}\n'.format(self.initial[0], self.initial[1]) \
#               + 'final : {} {}\n'.format(self.final[0], self.final[1])

    @property
    def final_safe(self):
        if self._final is None:
            self._final = deepcopy(self.final)
            for state in self._final:
                for mol in state.get_molecules():
                    mol.cell_state = self.cell_states[mol]
                    mol.set_state(state)

        return self._final

    @property
    def initial_safe(self):
        if self._cell_increment is None:
            return self.initial

        initial_copy = deepcopy(self.initial)
        for state in initial_copy[1:]:
            for mol in state.get_molecules():
                mol.cell_state = self.cell_states[mol]
                mol.set_state(state)

        initial_copy[1].cell_state = np.array(initial_copy[0].cell_state) - np.array(self._cell_increment)
        return initial_copy

    def set_cell_increment(self, cell_increment):
        self._cell_increment = cell_increment

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, state_list):
        self._transition_connect = None
        self._transport_connect = None
        self._initial = state_list

    @property
    def final(self):
        return self._final_test

    @final.setter
    def final(self, state_list):
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
        for state in self.final:
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
        for state in self.final:
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
        transition_donor = (self.initial[0], self.final_safe[1])
        transition_acceptor = (self.initial[1], self.final_safe[0])

        donor_vib_dos = self.vibrations.get_vib_spectrum(*transition_donor)  # (transition_donor)
        acceptor_vib_dos = self.vibrations.get_vib_spectrum(*transition_acceptor)  # (transition_acceptor)

        # print(donor_vib_dos)
        info = (hash(donor_vib_dos), hash(acceptor_vib_dos))

        # the memory is used if the overlap has been already computed (buggy)
        #if info in overlap_data:
        #    return overlap_data[info]

        def overlap(x):
            return donor_vib_dos(x) * acceptor_vib_dos(x)

        d = BOLTZMANN_CONSTANT*10000
        sup_range = np.abs(self.initial[0].energy - self.initial[1].energy) + d

        #if isinstance(self.vibrations, NoVibration):
        #    # Treat donor_vib_dos & acceptor_vib_dos as delta functions
        #    overlap_data[info] = 1.0 if donor_vib_dos - acceptor_vib_dos == 0 else 0.0
        #else:
        #    overlap_data[info] = quad(overlap, inf_range, sup_range, epsabs=1e-5, limit=1000)[0]

        overlap_data[info] = quad(overlap, -sup_range, sup_range, epsabs=1e-5, limit=1000)[0]

        return overlap_data[info]

    def get_electronic_coupling(self):
        return self._coupling_function(self.initial, self.final_safe, **self.arguments)

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
        return self.rate_function(self.initial, self.final_safe, **self.arguments)


class SimpleRate(BaseProcess):
    def __init__(self,
                 initial_states,
                 final_states,
                 rate_constant,
                 description='',
                 arguments=None
                 ):

        self._rate_constant = float(rate_constant)
        BaseProcess.__init__(self, initial_states, final_states, description, arguments)

    def get_rate_constant(self):
        return self._rate_constant


class SimpleRateMulti(BaseProcess):
    def __init__(self,
                 initial_states,
                 final_states,
                 rate_constant,
                 states_data,
                 description='',
                 ):

        # check if states_data contains proper keys
        assert 'energies' in states_data and 'probabilities' in states_data
        assert len(states_data['energies']) == len(states_data['probabilities'])

        self._states_data = states_data
        self._rate_constant = float(rate_constant)
        BaseProcess.__init__(self, initial_states, final_states, description, None)

    def get_rate_constant(self):

        # change final state energy randomly according to probabilities
        choice_state = np.random.choice(self._states_data['energies'], 1, p=self._states_data['probabilities'])

        self.final[0].modify_energy(choice_state[0])

        return self._rate_constant


class SimpleRateBounded(BaseProcess):
    def __init__(self,
                 initial_states,
                 final_states,
                 rate_constant,
                 range_distance,
                 step_param=1.0,
                 description='',
                 arguments=None
                 ):
        """
        :param initial_states:
        :param final_states:
        :param rate_constant:
        :param range_distance: max
        :param step_param: derivative of sigmoid function at range_distance
        :param description:
        :param arguments:
        """

        self._range_distance = range_distance
        self._rate_constant = rate_constant
        self._step_param = step_param
        BaseProcess.__init__(self, initial_states, final_states, description, arguments)

    def get_rate_constant(self):
        r_vector = np.array(self.initial_safe[0].get_coordinates_absolute()) - \
                   np.array(self.initial_safe[1].get_coordinates_absolute())
        distance = np.linalg.norm(r_vector)

        def sigmoid_function(x, distance, step_param=1.0):
            return 1.0 / (1.0 + np.exp(1.0 * step_param * 4 * (x - distance)))

        return self._rate_constant * sigmoid_function(distance, self._range_distance, step_param=self._step_param)


class InternalConversion(BaseProcess):
    def __init__(self,
                 initial_state,
                 final_state,
                 coupling=None,
                 vibrations=NoVibration(),
                 k1=0,
                 description='',
                 arguments=None
                 ):

        self._k1 = k1
        self._coupling = coupling
        self._vibrations = vibrations

        BaseProcess.__init__(self, [initial_state], [final_state], description, arguments)

    def get_rate_constant(self):
        # delta_g = self.final_test[0].energy - self.initial[0].energy

        return self._k1 + 2*np.pi/HBAR_PLANCK * self._coupling**2 * self.get_fcwd()

    @property
    def vibrations(self):
        return self._vibrations

    def get_fcwd(self):
        transition = (self.final_safe[0], self.initial[0])

        vib_dos = self.vibrations.get_vib_spectrum(*transition)

        #if isinstance(self.vibrations, NoVibration):
        #    return 1.0 if vib_dos == 0 else 0.0

        return vib_dos(0)


class MillerAbrahams(BaseProcess):
    # On development
    def __init__(self,
                 initial_state,
                 final_state,
                 attempt_to_hop=1,
                 inverse_localization_length=1,
                 vibrations=NoVibration(),
                 description='',
                 arguments=None
                 ):

        self._v0 = attempt_to_hop
        self._gamma = inverse_localization_length
        self._vibrations = vibrations

        BaseProcess.__init__(self, [initial_state], [final_state], description, arguments)

    def get_rate_constant(self):

        transition = (self.initial[0], self.final_safe[1])

        r_vector = np.array(self.initial_safe[0].get_coordinates_absolute()) - \
                   np.array(self.initial_safe[1].get_coordinates_absolute())
        distance = np.linalg.norm(r_vector)

        rate = self._v0 * np.exp(-2*self._gamma * distance)
        if transition[0].energy < transition[1].energy:
            rate = rate * self.vibrations.get_vib_spectrum(*transition)

        return rate

    @property
    def vibrations(self):
        return self._vibrations


class DecayRate(BaseProcess):
    def __init__(self,
                 initial_state,
                 final_state,
                 decay_rate_function,
                 description='',
                 arguments=None
                 ):

        BaseProcess.__init__(self, [initial_state], [final_state], description, arguments)
        self.rate_function = decay_rate_function

    def get_rate_constant(self):
        return self.rate_function(self.initial, self.final_safe, **self.arguments)
