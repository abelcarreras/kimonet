"""
File with conversion functions. All functions are analogous. They make a unit conversion from a physical
unit to the respective dimensionless atomic unit.

Each function needs two arguments:
        :param value: The value which has to be conversed
        :param d: Strig parameter that indicates whether the conversion is direct, for convention from
        the physical unit system to the atomic unit system , or inverse.
        :return The conversed value
"""


def from_ev_to_au(energy, d):
    if d is 'direct':
        return energy / 27.211

    if d is 'inverse':
        return energy * 27.211


def from_ns_to_au(time, d):
    if d is 'direct':
        return time / (2.4189 * 10**(-8))

    if d is 'inverse':
        return time * 2.4189 * 10**(-8)



def from_nm_to_au(length, d):
    if d is 'direct':
        return length / 0.053

    if d is 'inverse':
        return length * 0.053
