import numpy as np
from kimonet.conversion_functions import from_ev_to_au, from_ns_to_au
from kimonet.utils import rotate_vector
import copy
from collections import namedtuple


class Molecule:

    def __init__(self,
                 state_energies,
                 reorganization_energies,
                 transition_moment,
                 state='gs',
                 characteristic_length=10e-8,
                 coordinates=(0,),
                 orientation=(0, 0, 0)):                            # ax, ay, az
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

        :param characteristic_length: We consider a finite size molecule. The simplified shape of the molecule
        is longitudinal, squared or cubic and is defined with this characteristic length. Units: nm

        :param coordinates: 3d vector. Gives the position of the molecule in the system (in general the 0 position
        will coincide with the center of the distribution). Units: nm. If the system has less than 3 dimensions,
        the extra coordinates will be taken as 0.
        :param orientation: 3d unit vector. Gives the orientation of the molecule in the global reference system.
        """

        self.state_energies = state_energies
        self.state = state
        self.reorganization_energies = reorganization_energies
        self.transition_moment = np.array(transition_moment)
        self.characteristic_length = characteristic_length
        self.coordinates = np.array(coordinates)
        self.orientation = np.array(orientation)            # rotX, rotY, rotZ
        self.cell_state = np.zeros_like(coordinates, dtype=int)

    def __hash__(self):
        return hash((str(self.state_energies),
                     self.state,
                     str(self.reorganization_energies),
                     self.coordinates.tostring(),
                     self.orientation.tostring()))

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

    def get_reorganization_state_energy(self):
        return self.reorganization_energies[self.state]

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

        # Decay tuple
        # final: final state after dacay
        # description: any string with the description the decay
        Decay = namedtuple("Decay", ["final", "description"])

        decay_rates = {}

        if self.state == 's1':

            desexcitation_energy = self.state_energies[self.state] - self.state_energies['gs']      # energy in eV
            desexcitation_energy = from_ev_to_au(desexcitation_energy, 'direct')                    # energy in a.u.

            u = np.linalg.norm(self.transition_moment)              # transition moment norm.
            c = 137                                                 # light speed in atomic units

            rate = 4 * desexcitation_energy**3 * u**2 / (3 * c**3)

            decay_process = Decay(final='gs', description='singlet_radiative_decay')
            decay_rates[decay_process] = from_ns_to_au(rate, 'direct')

            # Example of second decay
            # -----------------------
            # decay_process = Decay(final='s2', description='test')
            # decay_rates[decay_process] = from_ns_to_au(rate, 'direct')

        # Example of decay in another state
        # ---------------------------------
        # if self.state == 's2':
        #     decay_process = Decay(final='s1', description='test2')
        #     decay_rates[decay_process] = from_ns_to_au(1000000, 'direct')

        return decay_rates

    def get_transition_moment(self):
        return rotate_vector(self.transition_moment, self.orientation)

    def copy(self):
        return copy.deepcopy(self)

    def get_orientation_vector(self):
        return rotate_vector([1, 0, 0][:self.get_dim()], self.orientation)