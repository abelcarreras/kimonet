import copy


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

    def add_molecule(self, molecule):
        if not molecule in self._molecules_set:
            self._molecules_set.append(molecule)

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


ground_state = State(label='gs', energy=0.0, multiplicity=1)


if __name__ == '__main__':
    print('Test state')
    s = State(label='s1', energy=1.5, multiplicity=1)
    print(hash(s))