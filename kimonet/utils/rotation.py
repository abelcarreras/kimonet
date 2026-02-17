import numpy as np


def rotate_vector(vector, orientation):
    """Rotate a vector of dimension 1-3 using effective Euler angles

    :param vector: vector to be rotated
    :param orientation: orientation list (1D: only rotation in X, 2D: only rotation in Z, 3D: Euler angles (RX RY RZ))

    :return: rotated vector

    """

    n_dim = len(vector)
    norm = np.linalg.norm(vector)
    rotated = vector.copy()

    if n_dim == 1:
        # 1D: flip sign if angle > pi/2
        if orientation[0] % (2*np.pi) > np.pi/2:
            rotated = -np.array(rotated)

    elif n_dim == 2:
        ang_z = orientation[2]
        c, s = np.cos(ang_z), np.sin(ang_z)
        R = np.array([[c, -s],
                      [s,  c]])
        rotated = R @ vector

    elif n_dim == 3:
        ang_x, ang_y, ang_z = orientation
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ang_x), -np.sin(ang_x)],
                       [0, np.sin(ang_x),  np.cos(ang_x)]])
        Ry = np.array([[ np.cos(ang_y), 0, np.sin(ang_y)],
                       [0, 1, 0],
                       [-np.sin(ang_y), 0, np.cos(ang_y)]])
        Rz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0],
                       [np.sin(ang_z),  np.cos(ang_z), 0],
                       [0, 0, 1]])
        rotated = Rz @ Ry @ Rx @ vector

    # normalize
    return rotated / np.linalg.norm(rotated) * norm


def get_orientation_from_vectors(v_reference, v_target):
    """
    get orientation angles that rotate a v_reference vector to a v_target vector

    :param v_reference: reference vector
    :param v_target: target vector
    :return: orientation angles
    """

    # Compute Z-Y-X Euler angles to rotate v_init to v_final
    v_reference = np.array(v_reference, dtype=float)/np.linalg.norm(v_reference)
    v_target = np.array(v_target, dtype=float) / np.linalg.norm(v_target)

    assert len(v_reference) == len(v_target)

    n_dim = len(v_reference)

    if n_dim == 1:
        if n_dim == 1:
            v_reference = v_reference / np.linalg.norm(v_reference)
            v_target = v_target / np.linalg.norm(v_target)

            # If vectors have same direction -> no rotation
            if np.dot(v_reference, v_target) > 0:
                angle = 0.0
            else:
                # Opposite direction -> flip (pi rotation)
                angle = np.pi

            return [angle, 0.0, 0.0]

    if n_dim == 2:
        angle = np.arctan2(v_reference[0] * v_target[1] - v_reference[1] * v_target[0],  # cross product (z-component)
                           np.dot(v_reference, v_target))  # dot product

        return np.array([0.0, 0.0, angle]).tolist()

    elif n_dim == 3:
        # Compute rotation axis and angle
        axis = np.cross(v_reference, v_target)
        sin_angle = np.linalg.norm(axis)
        cos_angle = np.dot(v_reference, v_target)

        if sin_angle < 1e-8:
            # Vectors are colinear: no rotation or 180 deg
            if cos_angle > 0:
                return 0.0, 0.0, 0.0  # no rotation
            else:
                # 180 deg rotation around any perpendicular axis
                axis = np.array([1, 0, 0]) if abs(v_reference[0]) < 0.9 else np.array([0, 1, 0])
                axis /= np.linalg.norm(axis)
                angle = np.pi
        else:
            axis /= sin_angle
            angle = np.arctan2(sin_angle, cos_angle)

        # Rodrigues formula to get rotation matrix
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # Decompose R into Z-Y-X Euler angles
        if abs(R[2, 0]) < 1.0:
            ang_y = -np.arcsin(R[2, 0])
            ang_x = np.arctan2(R[2, 1]/np.cos(ang_y), R[2, 2]/np.cos(ang_y))
            ang_z = np.arctan2(R[1, 0]/np.cos(ang_y), R[0, 0]/np.cos(ang_y))
        else:
            # Gimbal lock: ang_y = +/- pi/2
            ang_y = np.pi/2 if R[2, 0] <= -1 else -np.pi/2
            ang_x = 0
            ang_z = np.arctan2(-R[0, 1], R[1, 1])

        return np.array([ang_x, ang_y, ang_z]).tolist()

    raise Exception('Wrong dimension, must be 1, 2 or 3')


if __name__ == '__main__':
    orientation = get_orientation_from_vectors([5, 0, 0], [1.0, -1.0, 0.0])
    print('3D orientation: ', orientation)
    vector = rotate_vector([5, 0, 0], orientation)
    print('3D result: ', vector, np.linalg.norm(vector))

    orientation = get_orientation_from_vectors([5, 0], [-2.0, 1.0])
    print('2D orientation: ', orientation)
    vector = rotate_vector([5, 0], orientation)
    print('2D result: ', vector, np.linalg.norm(vector))

    orientation = get_orientation_from_vectors([5], [-1.0])
    print('1D orientation: ', orientation)
    vector = rotate_vector([5], orientation)
    print('1D result: ', vector, np.linalg.norm(vector))

