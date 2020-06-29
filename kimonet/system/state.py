
class State:
    def __init__(self,
                 label,
                 energy,
                 multiplicity=1):
        self._label = label
        self._energy = energy
        self._multiplicity=multiplicity

    @property
    def label(self):
        return self._label

    @property
    def energy(self):
        return self._energy

    @property
    def multiplicity(self):
        return self._multiplicity
