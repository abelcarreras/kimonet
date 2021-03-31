import numpy as np
from kimonet.utils import rotate_vector
import copy
from kimonet.system.state import ground_state as _GS_


class Molecule(object):

    def __init__(self,
                 name=None,
                 state=None,
                 vdw_radius=1.0,  # Angstrom
                 coordinates=(0,),  # Angstrom
                 orientation=(0, 0, 0),  # Rx, Ry, Rz (radians)
                 site_energy=0.0 # eV
                 ):
        """
        :param states_energies: dictionary {'state': energy} (eV)
        :param state: string containing the current state
        :param coordinates: the coordinates vector of the molecule within the system (Angstrom)
        :param orientation: 3d unit vector containing the orientation angles of the molecule defined in radiants respect X, Y and Z axes.
        """

        # set state energies to vibrations
        #vibrations.set_state_energies(state_energies)

        self._coordinates = np.array(coordinates)
        self.orientation = np.array(orientation)
        self._cell_state = np.zeros_like(coordinates, dtype=int)
        self._vdw_radius = vdw_radius
        self._site_energy = site_energy
        self.name = name

        if state is None:
            self._state = _GS_.copy()
        else:
            self._state = state

        self._state.remove_molecules()
        self._state.add_molecule(self)

    def __hash__(self):
        return hash((
                     # self._state.label,
                     self.name,
                     self._coordinates.tobytes(),
                     self.orientation.tobytes()
                     # np.array2string(self._cell_state, precision=12)
                     )
                    )

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_vdw_radius(self):
        return self._vdw_radius

    def get_dim(self):
        return len(self._coordinates)

    def set_coordinates(self, coordinates):
        """
        sets the coordinates of the molecule
        :param coordinates: coordinate vector
        """
        self._coordinates = np.array(coordinates)
        self._cell_state = np.zeros_like(self._coordinates, dtype=int)

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

    @property
    def cell_state(self):
        return self._cell_state

    @cell_state.setter
    def cell_state(self, c_state):
        self._cell_state = np.array(c_state)

    @property
    def cell_state_2(self):
        return self._cell_state + self.state.cell_state

    @property
    def site_energy(self):
        return self._site_energy
