import copy
import math

# import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
# import scipy
# import scipy.interpolate

import mmoi_utils


class Environment:
    def __init__(self, g=9.80665, air_density=1.225, kinematic_viscosity=1.81e-5):
        self.__g = g
        self.__air_density = air_density
        self.__kinematic_viscosity = kinematic_viscosity

    @property
    def g(self):
        return self.__g

    @property
    def air_density(self):
        return self.__air_density

    @property
    def kinematic_viscosity(self):
        return self.__kinematic_viscosity


class Aero:
    @classmethod
    def ReynoldsNumber(cls, env, length, velocity):
        return env.air_density * velocity * length / env.kinematic_viscosity


class Material:
    def __init__(self, name, density):
        self.__name = name
        self.__density = density

    @property
    def name(self): return self.__name

    @property
    def density(self): return self.__density
    # cardboard = rocket_components.Material('Cardboard', 1.0)  # TODO: SET THE CORRECT MATERIAL
    # foam = rocket_components.Material('Readiboard', 1.0)  # TODO: SET THE CORRECT MATERIAL

# TODO: MAKE A MATERIAL DIRECTORY... LIKE THE ENGINE DIRECTORY


class Component:
    __id_next = 0

    def __init__(self, pos=np.array([0.0, 0.0, 0.0]), mass=0.0, children=None, offset=None, calc_aero=False):
        self.__id = Component.__id_next
        Component.__id_next += 1
        children = [] if children is None else children
        self.__pos = pos
        self.__offset = np.array([0, 0, 0]) if offset is None else offset
        self.__mass = mass
        self.__parent = None
        self.__children = children
        for child in children:
            child.parent = self
        self.__calc_aero = calc_aero

    @property
    def id(self): return self.__id

    def add(self, child):
        self.__children.append(child)
        child.parent = self

    @property
    def abs_pos(self):
        if self.__parent is None:
            return self.__pos
        return self.__parent.abs_pos + self.__pos

    @abs_pos.setter
    def abs_pos(self, p):
        if self.__parent is None:
            self.__pos = p
        else:
            self.__pos = p - self.__parent.abs_pos
        return p

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, p):
        self.__pos = p - self.__offset
        return self.__pos

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, v):
        self.__offset = v
        # Note: this might not be best
        return self.__offset

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, p):
        if p is None:
            abs_pos = self.abs_pos
            self.__parent = None
            self.__pos = abs_pos
        return self.__parent

    @property
    def children(self):
        return self.__children

    def mass(self, t0):
        child_masses = [child.mass(t0) for child in self.__children]
        return self.calc_mass(t0) + (sum(child_masses) if len(child_masses) > 0 else 0)

    def cg(self, t0):
        child_contributions = [child.mass(t0) * child.cg(t0) for child in self.__children]
        self_contribution = self.calc_mass(t0) * self.abs_pos
        cg_sum = self_contribution
        for child_contribution in child_contributions:
            cg_sum += child_contribution
        return cg_sum / self.mass(t0)

    def mmoi(self, t0):
        mmoi_sum = np.zeros(3)
        self_cg = self.cg(t0)
        for child in self.__children:
            mmoi_sum += child.mmoi(t0)
            offset = child.cg(t0) - self_cg
            mmoi_sum += mmoi_utils.parallel_axis_components(child.mass(t0), offset=offset)
        mmoi_sum += self.calc_mmoi_at_cg(t0)
        mmoi_sum += mmoi_utils.parallel_axis_components(self.__mass, offset=self_cg-self.pos)
        return mmoi_sum

    def calc_mass(self, t0):
        return self.__mass

    def calc_mmoi_at_cg(self, t0):
        # Returns the component MMOI for (x, y, z) axes
        return np.zeros(3)

    def drag(self, v, env):
        total_drag = self.calc_drag(v, env)
        for child in self.__children:
            total_drag += child.drag(v, env)
        return total_drag

    def calc_drag(self, v, env):  # NOTE: THIS SHOULD GIVE DRAG IN THREE DIMENSIONS
        if self.__calc_aero:
            return 0.5 * env.air_density * self.frontal_area * (v**2) * self.calc_cd(v, env)
        return 0

    def calc_cd(self, v, env):
        return 0.35

    def duplicate(self):
        dup = copy.copy(self)
        dup.__id = Component.__id_next
        Component.__id_next += 1
        return dup


class EmptyComponent(Component):
    def __init__(self, children=None):
        children = [] if children is None else children
        Component.__init__(self, np.array([0.0, 0.0, 0.0]), 0.0, children=children)


class PhysicalComponent(Component):
    def __init__(self, pos, mass, material=None, children=None, mass_override=False, calc_aero=False):
        children = [] if children is None else children
        Component.__init__(self, pos, mass, children=children, calc_aero=calc_aero)
        self.__material = material
        self.__mass_override = mass_override

    @property
    def material(self):
        return self.__material


class TubeComponent(PhysicalComponent):
    def __init__(self, L, id, od, material, mass, pos=None, children=None, calc_aero=True):
        children = [] if children is None else children
        pos = np.array([L/2, 0.0, 0.0]) if pos is None else pos
        self.__length = L
        self.__inner_diameter = id
        self.__outer_diameter = od
        self.__thickness = (od-id)/2
        self.__volume = L * math.pi * (od*self.thickness - (self.thickness**2))
        self.__frontal_area = math.pi * ((od/2)**2)
        self.__surface_area = math.pi * od * L
        mass_override = mass is not None
        if not mass_override:
            mass = self.__volume * material.density
        PhysicalComponent.__init__(self, pos, mass, material=material, children=children, mass_override=mass_override, calc_aero=calc_aero)

    @property
    def length(self):
        return self.__length

    @property
    def inner_diameter(self):
        return self.__inner_diameter

    @property
    def outer_diameter(self):
        return self.__outer_diameter

    @property
    def thickness(self):
        return self.__thickness

    @property
    def volume(self):
        return self.__volume

    @property
    def frontal_area(self):
        return self.__frontal_area

    @property
    def surface_area(self):
        return self.__surface_area

    def calc_mmoi_at_cg(self, t0):
        return mmoi_utils.cylindrical_shell(self.length, self.outer_diameter/2, self.mass(t0))

    def calc_drag(self, v):
        # Calculate CD(0) = Cd_fb + Cd_b ( not fins or interference)
        # Use Cf ( using the Reynolds number )
        return 0


class AeroBody(Component):  # TODO: THIS NEEDS TO BE REMOVED...
    def __init__(self, L, d, mass, cd, children=None, pos=np.array([0.0, 0.0, 0.0])):
        children = [] if children is None else children
        Component.__init__(self, pos, mass, children=children, calc_aero=True)
        self.__L = L
        self.__d = d
        self.__cd = cd
        self.__area = math.pi * ((d/2.0)**2)
        self.__rho = 1.2754

    def calc_drag(self, v, env):
        return 0.5 * env.air_density * self.__area * (v**2) * self.__cd
