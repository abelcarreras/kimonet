

class Transition:
    def __init__(self, state1, state2, symmetric=True, tdm=0, reorganization_energy=0):
        self._state1 = state1
        self._state2 = state2
        self._symmetric = symmetric
        self._tdm = tdm
        self._reorganization_energy = reorganization_energy

    def __str__(self):
        return '{} -> {}'.format(self._state1.label, self._state2.label)

    def __hash__(self):
        if self._symmetric:
            return hash(self._state1.label) + hash(self._state2.label)
        else:
            return hash((self._state1.label, self._state2.label))

    def __eq__(self, other):
        if (self._state1.label, self._state2.label) == (other._state1.label, other._state2.label):
            return True
        if self._symmetric or other._symmetric:
            if (self._state1.label, self._state2.label) == (other._state2.label, other._state1.label):
                return True

        return hash(self) == hash(other)

    @property
    def tdm(self):
        return self._tdm

    @property
    def reorganization_energy(self):
        return self._reorganization_energy