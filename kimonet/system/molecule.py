import numpy as np
from kimonet.utils import rotate_vector
import copy
from kimonet.utils.units import DEBYE_TO_ANGS_EL

_ground_state_ = 'gs'


class Molecule:

    def __init__(self,
                 state_energies,                # eV
                 reorganization_energies,       # eV
                 transition_moment,             # Debye
                 decays=None,
                 state='gs',
                 vdw_radius=1.0,                # Angstrom
                 coordinates=(0,),              # Angstrom
                 orientation=(0, 0, 0),        # Rx, Ry, Rz (radians)
                 vib_spectrum=None):
        """
        :param states_energies: dictionary {'state': energy}
        :param state: sting of the name of the state
        The name of the state should coincide with some key of the dictionary in order to identify the state with
        its energy.
        :param reorganization_energies: dictionary {'state': relaxation energy of the state}
        Names of 'state' would be: g_s (ground state), s_1 (first singlet), t_1 (first triplet), etc.
        Energies should be given with eV.

        :param transition_moment: Dipole transition moment vector (3d). The vector is given in respect to the RS
        of the molecule. So for all molecules of a same type if will be equal.
        This dipole moment is given in atomic units.

        :param coordinates: 3d vector. Gives the position of the molecule in the system (in general the 0 position
        will coincide with the center of the distribution). Units: nm. If the system has less than 3 dimensions,
        the extra coordinates will be taken as 0.
        :param orientation: 3d unit vector. Gives the orientation of the molecule in the global reference system.
        """

        self.state = state
        self.state_energies = state_energies
        self.reorganization_energies = reorganization_energies
        self.coordinates = np.array(coordinates)
        self.orientation = np.array(orientation)
        self.cell_state = np.zeros_like(coordinates, dtype=int)
        self.vdw_radius = vdw_radius
        self.vib_spectrum = {} if vib_spectrum is None else vib_spectrum

        self.transition_moment = {}
        for k, v in transition_moment.items():
            self.transition_moment[k] = np.array(v) * DEBYE_TO_ANGS_EL
        # self.transition_moment = np.array(transition_moment) * DEBYE_TO_ANGS_EL  # Debye -> Angs * e


        self.decays = {} if decays is None else decays
        self.decay_dict = {}

    def __hash__(self):
        return hash((str(self.state_energies),
                     self.state,
                     str(self.reorganization_energies),
                     self.coordinates.tostring(),
                     self.orientation.tostring()))

    def get_vdw_radius(self):
        return self.vdw_radius

    def get_dim(self):
        return len(self.coordinates)

    def set_coordinates(self, coordinate_list):
        """
        :param coordinate_list: List [x, y, z] with the coordinates of the molecule. Units: nm
        Changes self.coordinates to this new position. Format: numpy array.
        """
        self.coordinates = np.array(coordinate_list)
        self.cell_state = np.zeros_like(self.coordinates, dtype=int)

    def get_coordinates(self):
        """
        :return: Array with the molecular coordinates.
        """
        return self.coordinates

    def set_orientation(self, orientation):
        """
        :param orientation: list with the coordinates of the orientation vector
        Changes self.orientation to this new orientation. Format: numpy array
        """
        self.orientation = np.array(orientation)

    def molecular_orientation(self):
        """
        :return: Array with the molecular orientation
        """
        return self.orientation

    def get_reorganization_energy(self, state=None):
        if state is None:
            return self.reorganization_energies[self.state]
        else:
            return self.reorganization_energies[state]

    def get_state_energy(self, state=None):
        if state is None:
            return self.state_energies[self.state]
        else:
            return self.state_energies[state]

    def get_vib_spectrum(self, transition):
        if transition in self.vib_spectrum:
            return self.vib_spectrum[transition]
        else:
            return None

    def electronic_state(self):
        """
        :return: the electronic state of the molecule
        """
        return self.state

    def set_state(self, new_state):
        """
        :param new_state:
        No return method. Only changes the molecular state when the exciton is transferred.
        """
        self.state = new_state

    def desexcitation_energies(self):
        """
        IS NOT USED (19/08/2019).
        Given an electronic state, calculates the possible desexcitation energy. Generates and sorts
        a list with the energies, then calculates the possible desexcitation energies (the energy difference
        between the given state and the less energetic states).
        :return: Dictionary with the decay processes as key, e.g. 'State1_to_State0', and the energy as argument
        """
        desexcitations = {}

        for state_key in self.state_energies:
            if self.state_energies[self.state] > self.state_energies[state_key]:
                decay_process = 'from_'+self.state+'_to_'+state_key
                energy_gap = self.state_energies[self.state] - self.state_energies[state_key]
                desexcitations[decay_process] = energy_gap

        return desexcitations

    def decay_rates(self):
        """
        :return: A list of two elements: list of the possible decay processes and another with the respective rates
        for a given electronic state.

        More if(s) entrances shall be added if more electronic states are considered.
        """

        if self.state in self.decay_dict:
            return self.decay_dict[self.state]

        decay_rates = {}
        for coupling in self.decays:
            if coupling.initial == self.state:
                decay_rates[coupling] = self.decays[coupling](self)

        self.decay_dict[self.state] = decay_rates

        return decay_rates

    def get_transition_moment(self, to_state=_ground_state_):
        if (self.state, to_state) in self.transition_moment:
            return rotate_vector(self.transition_moment[(self.state, to_state)], self.orientation)
        elif (to_state, self.state) in self.transition_moment:
            return rotate_vector(self.transition_moment[(to_state, self.state)], self.orientation)
        else:
            return np.zeros(self.get_dim())

    def copy(self):
        return copy.deepcopy(self)

    def get_orientation_vector(self):
        return rotate_vector([1, 0, 0][:self.get_dim()], self.orientation)