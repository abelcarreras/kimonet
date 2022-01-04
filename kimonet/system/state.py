import copy
import numpy as np


class State(object):
    def __init__(self,
                 label,
                 energy,
                 multiplicity=1,
                 size=1,
                 molecules_list=None,
                 connected_distance=1):

        self._label = label
        self._energy = energy
        self._multiplicity = multiplicity
        self._size = size
        self._connected_distance = connected_distance
        self._molecules_set = molecules_list if molecules_list is not None else []
        self._cell_state = None
        self.supercell = None

    def __hash__(self):
        return hash((self._label,
                     self._energy,
                     self._multiplicity,
                     self._size,
                     str(self._cell_state) if self._label != ground_state.label else '',
                     tuple(self.get_molecules())))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def copy(self):
        return copy.deepcopy(self)

    def get_molecules(self):
        assert self._molecules_set is not None
        return self._molecules_set

    def get_center(self):
        assert self._molecules_set is not None
        return self._molecules_set[0]

    def get_coordinates(self):
        # print(self.get_center().get_coordinates())
        return np.average([mol.get_coordinates() for mol in self.get_molecules()], axis=0)

    def get_coordinates_relative(self, supercell):

        supercell = np.array(supercell)
        coor_average = []
        for mol in self.get_molecules():
            coor_average.append(mol.get_coordinates() - np.dot(supercell.T, mol.cell_state))

        return np.average(coor_average, axis=0)

    def get_coordinates_absolute(self):

        supercell = np.array(self.supercell)
        coor_average = []
        for mol in self.get_molecules():
            coor_average.append(mol.get_coordinates() - np.dot(supercell.T, mol.cell_state + self.cell_state))

        return np.average(coor_average, axis=0)

    def add_molecule(self, molecule):
        if not molecule in self._molecules_set:
            self._molecules_set.append(molecule)
            if self._cell_state is None:
                self._cell_state = np.zeros_like(molecule.get_coordinates(), dtype=int)

    def reorganize_cell(self):

        cell_diff = np.array(np.average([np.array(mol.cell_state) for mol in self.get_molecules()], axis=0), dtype=int)

        for mol in self.get_molecules():
            mol.cell_state = np.array(mol.cell_state) - cell_diff

        self.cell_state += cell_diff

    def remove_molecules(self):
        self._molecules_set = []
        # self._cell_state = None

    def reset_molecules(self):
        for mol in self.get_molecules():
            # mol.set_state(self)
            mol.cell_state *= 0

    def modify_energy(self, energy):
        self._energy = energy

    @property
    def label(self):
        return self._label

    @property
    def energy(self):
        energy = self._energy
        for mol in self.get_molecules():
            energy += mol.site_energy
        return energy

    @property
    def multiplicity(self):
        return self._multiplicity

    @property
    def size(self):
        return self._size

    @property
    def connected_distance(self):
        return self._connected_distance

    @property
    def cell_state(self):
        assert self._cell_state is not None
        return self._cell_state

    @cell_state.setter
    def cell_state(self, c_state):
        self._cell_state = np.array(c_state)

    @property
    def vdw_radius(self):
        return np.average([mol.get_vdw_radius() for mol in self.get_molecules()])


# definition of ground state
ground_state = State(label='gs', energy=0.0, multiplicity=1)

if __name__ == '__main__':
    print('Test state')
    s = State(label='s1', energy=1.5, multiplicity=1)
    print(hash(s))
    s2 = State(label='s1', energy=1.5, multiplicity=1)
    s2.cell_state = [1, 2]
    print(hash(s2))

    print(s == s2)
    print(s in [s])
    print(s2 in [s])