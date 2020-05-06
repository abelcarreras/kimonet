import numpy as np
from kimonet.utils import rotate_vector
import copy
from kimonet.utils.units import DEBYE_TO_ANGS_EL
from kimonet.system.vibrations import NoVibration
from kimonet import _ground_state_


class Molecule:

    def __init__(self,
                 state_energies,  # eV
                 transition_moment,  # Debye
                 vibrations=NoVibration(),
                 name=None,
                 decays=None,
                 state=_ground_state_,
                 vdw_radius=1.0,  # Angstrom
                 coordinates=(0,),  # Angstrom
                 orientation=(0, 0, 0),  # Rx, Ry, Rz (radians)
                 ):
        """
        :param states_energies: dictionary {'state': energy} (eV)
        :param state: string containing the current state
        :param transition_moment: Transition dipole moment dictionary (Debye)
        :param vibrations: Vibrations object. This contains all the information about how to handle temperature dependence
        :param coordinates: the coordinates vector of the molecule within the system (Angstrom)
        :param orientation: 3d unit vector containing the orientation angles of the molecule defined in radiants respect X, Y and Z axes.
        """

        # set state energies to vibrations
        vibrations.set_state_energies(state_energies)

        self.state = state
        self.state_energies = state_energies
        self._coordinates = np.array(coordinates)
        self.orientation = np.array(orientation)
        self.cell_state = np.zeros_like(coordinates, dtype=int)
        self.vdw_radius = vdw_radius
        self.vibrations = vibrations
        self.name = name

        self.transition_moment = {}
        for k, v in transition_moment.items():
            self.transition_moment[k] = np.array(v) * DEBYE_TO_ANGS_EL
        # self.transition_moment = np.array(transition_moment) * DEBYE_TO_ANGS_EL  # Debye -> Angs * e

        self.decays = {} if decays is None else decays
        self.decay_dict = {}

    def __hash__(self):
        return hash((str(self.state_energies),
                     self.state,
                     # str(self.reorganization_energies),
                     self._coordinates.tostring(),
                     self.orientation.tostring())) + \
               hash(self.vibrations)

    def get_vdw_radius(self):
        return self.vdw_radius

    def get_dim(self):
        return len(self._coordinates)

    def set_coordinates(self, coordinates):
        """
        sets the coordinates of the molecule
        :param coordinates: coordinate vector
        """
        self._coordinates = np.array(coordinates)
        self.cell_state = np.zeros_like(self._coordinates, dtype=int)

    def get_coordinates(self):
        """
        sets the molecule coordinates
        :return: Array with the molecular coordinates.
        """
        return self._coordinates

    def set_orientation(self, orientation):
        """
        sets the orientation angles
        :param orientation: the orientation angles
        """
        self.orientation = np.array(orientation)

    def molecular_orientation(self):
        """
        :return: Array with the molecular orientation angles
        """
        return self.orientation

    def get_state_energy(self, state=None):
        if state is None:
            return self.state_energies[self.state]
        else:
            return self.state_energies[state]

    def get_vib_dos(self, transition, temperature=300):
        return self.vibrations.get_vib_spectrum(transition, temperature)

    def electronic_state(self):
        """
        :return: the electronic state of the molecule
        """
        return self.state

    def set_state(self, new_state):
        """
        :param new_state:
        Changes the molecular state of the molecule.
        """
        self.state = new_state

    def decay_rates(self):
        """
        returns the dacay rate for the current state
        :return: decay rate.

        """

        if self.state not in self.decay_dict:
            decay_rates = {}
            for coupling in self.decays:
                if coupling.initial == self.state:
                    decay_rates[coupling] = self.decays[coupling](self)

            self.decay_dict[self.state] = decay_rates

        return self.decay_dict[self.state]

    def get_transition_moment(self, to_state=_ground_state_):
        """
        returns the transition dipole moment between the current state and the requested state (by default ground state)
        :param to_state: the transition dipole moment is given between this state and the current state
        :return:
        """
        if (self.state, to_state) in self.transition_moment:
            return rotate_vector(self.transition_moment[(self.state, to_state)], self.orientation)
        elif (to_state, self.state) in self.transition_moment:
            return rotate_vector(self.transition_moment[(to_state, self.state)], self.orientation)
        else:
            return np.zeros(self.get_dim())

    def copy(self):
        """
        returns a deep copy of this molecule
        :return: a copy of molecule
        """
        return copy.deepcopy(self)

    def get_orientation_vector(self):
        """
        return a vector that indicates the main reference orientation axis of the molecule.
        All other vector properties of the molecule are defined respect the molecule orientation
        This vector does not define the orientation completely, just serves as visual reference
        :return:
        """
        return rotate_vector([1, 0, 0][:self.get_dim()], self.orientation)
