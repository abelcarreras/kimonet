

class Transition:
    def __init__(self, state1, state2, symmetric=True):
        self._state1 = state1
        self._state2 = state2
        self._symmetric = symmetric

    def __hash__(self):
        if self._symmetric:
            return hash(self._state1.label) + hash(self._state2.label)
        else:
            return hash((self._state1.label, self._state2.label))

    def __eq__(self, other):
        return hash(self) == hash(other)
