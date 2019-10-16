import numpy as np


def rot_x(angle):
    """
    rotate with respect x axis (right hand criteria)
    :param angle:
    :return:
    """
    return np.array([[1,             0,              0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle),  np.cos(angle)]])


def rot_y(angle):
    """
    rotate with respect y axis (right hand criteria)
    :param angle:
    :return:
    """
    return np.array([[np.cos(angle),  0, np.sin(angle)],
                     [0,              1,             0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def rot_z(angle):
    """
    rotate with respect z axis (right hand criteria)
    :param angle:
    :return:
    """
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0,                          0, 1]])


def rotate_vector_3d(vector, ang_x, ang_y, ang_z):
    return np.dot(rot_z(ang_z), np.dot(rot_y(ang_y), np.dot(rot_x(ang_x), vector)))


def rotate_vector(vector, orientation):
    """
    Rotate vector of dimensions (1-3)

    :param vector: vector to rotate
    :param orientation: angles (in radians) to rotate respect to axis x, y, z
    :return: rotated vector
    """

    norm = np.linalg.norm(vector)
    n_dim = len(vector)
    ang_x, ang_y, ang_z = orientation
    rotated_vector = np.dot(rot_z(ang_z)[:n_dim, :n_dim],
                        np.dot(rot_y(ang_y)[:n_dim, :n_dim],
                               np.dot(rot_x(ang_x)[:n_dim, :n_dim],
                                      vector)))

    # Rotated vector is defined to have same norm as original whatever orientation
    return rotated_vector/np.linalg.norm(rotated_vector) * norm
