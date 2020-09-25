import copy
import numpy as np


class State:
    def __init__(self,
                 label,
                 energy,
                 multiplicity=1,
                 size=1,
                 molecules_list=None):
        self._label = label
        self._energy = energy
        self._multiplicity = multiplicity
        self._size = size
        self._molecules_set = molecules_list if molecules_list is not None else []
        self._cell_state = None

    def __hash__(self):
        return hash((self._label, self._energy, self._multiplicity, self._size))

    def copy(self):
        return copy.deepcopy(self)

    def get_molecules(self):
        assert self._molecules_set is not None
        return self._molecules_set

    def get_center(self):
        assert self._molecules_set is not None
        return self._molecules_set[0]

    def get_coordinates(self):
        return self.get_center().get_coordinates()


    def add_molecule(self, molecule):
        if not molecule in self._molecules_set:
            self._molecules_set.append(molecule)
            if self._cell_state is None:
                self._cell_state = np.zeros_like(molecule.get_coordinates(), dtype=int)

    def remove_molecules(self):
        self._molecules_set = []
        self._cell_state = None

    @property
    def label(self):
        return self._label

    @property
    def energy(self):
        return self._energy

    @property
    def multiplicity(self):
        return self._multiplicity

    @property
    def size(self):
        return self._size

    @property
    def cell_state(self):
        assert self._cell_state is not None
        return self._cell_state

    @cell_state.setter
    def cell_state(self, c_state):
        self._cell_state = c_state


ground_state = State(label='gs', energy=0.0, multiplicity=1)


if __name__ == '__main__':
    print('Test state')
    s = State(label='s1', energy=1.5, multiplicity=1)
    print(hash(s))