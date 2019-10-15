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
                     [np.sin(angle), np.cos(angle), 0],
                     [0,                         0, 1]])


def rot_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])


def rot_2d_1(angle):
    return np.array([[1, 0],
                     [0,  np.cos(angle)]])


def rotate_vector_3d(vector, ang_x, ang_y, ang_z):
    return np.dot(rot_z(ang_z), np.dot(rot_y(ang_y), np.dot(rot_x(ang_x), vector)))


def rotate_vector_2d(vector, ang):
    return np.dot(rot_2d(ang), vector)


def rotate_vector(vector, orientation):

    norm = np.linalg.norm(vector)
    ndim = len(vector)
    ang_x, ang_y, ang_z = orientation
    new_vector = np.dot(rot_z(ang_z)[:ndim, :ndim],
                        np.dot(rot_y(ang_y)[:ndim, :ndim],
                               np.dot(rot_x(ang_x)[:ndim, :ndim],
                                      vector)))

    return new_vector/np.linalg.norm(new_vector) * norm
