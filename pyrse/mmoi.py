
import numpy as np
import numpy.linalg


def parallel_axis_components(mass, offset=None):
    if offset is None:
        return np.zeros(3)
    dx = np.linalg.norm(offset - np.dot(offset, np.array([1, 0, 0])) * np.array([1, 0, 0]))
    dy = np.linalg.norm(offset - np.dot(offset, np.array([0, 1, 0])) * np.array([0, 1, 0]))
    dz = np.linalg.norm(offset - np.dot(offset, np.array([0, 0, 1])) * np.array([0, 0, 1]))
    return (mass * np.array([dx**2, dy**2, dz**2]))


def point_mass(mass, offset=None, rotation=None):
    return parallel_axis_components(mass, offset=offset)


def slender_rod(length, mass, offset=None, rotation=None):
    """Rod length along the x axis"""
    I_yz = (1/12)*mass*(length**2)
    return np.array([0, I_yz, I_yz])


def cylinder(length, radius, mass, offset=None, rotation=None):
    """Cylinder length along the x axis"""
    I_yz = (1/12)*mass*(3*(radius**2) + (length**2))
    I_x = (1/2)*mass*(radius**2)
    return np.array([I_x, I_yz, I_yz])


def cylindrical_shell(length, radius, mass, offset=None, rotation=None):
    """Cylinder length along the x axis"""
    I_yz = (1/6)*mass*(3*(radius**2) + (length**2))
    I_x = mass*(radius**2)
    return np.array([I_x, I_yz, I_yz])


# Rectangular Prizm
# Right Cone
# Right Conic Shell
# Nose Cones (Conic, Parobolic, VanKarman, etc.)

if __name__ == '__main__':
    pass
