import numpy as np
from kimonet.utils import rotate_vector
import copy
from kimonet.utils.units import DEBYE_TO_ANGS_EL
from kimonet.system.vibrations import NoVibration
from kimonet.system.state import ground_state as _GS_
from kimonet.core.processes.types import Transition


class Molecule:

    def __init__(self,
                 name=None,
                 state=_GS_,
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
        #vibrations.set_state_energies(state_energies)

        # self._state = self._labels_to_state[state]
        self._state = state
        self._coordinates = np.array(coordinates)
        self.orientation = np.array(orientation)
        self.cell_state = np.zeros_like(coordinates, dtype=int)
        self.vdw_radius = vdw_radius
        #self.vibrations = vibrations
        self.name = name

        state.add_molecule(self)

    def __hash__(self):
        return hash((self._state,
                     # str(self.reorganization_energies),
                     np.array2string(self._coordinates, precision=12),
                     np.array2string(self.orientation, precision=12),
                     np.array2string(self.cell_state, precision=12)))

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

    def get_state_energy(self):
        return self._state.energy

#    def get_vib_dos(self, transition):
#        return self.vibrations.get_vib_spectrum(transition)

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

    def set_state(self, state):
        state.add_molecule(self)
        self._state = state

    @property
    def state(self):
        return self._state
