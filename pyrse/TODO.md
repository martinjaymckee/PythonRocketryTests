Coordinates:
- Add the ability to transform vectors (such as velocity) from the ECEF coordinates to ENU

General Rocket Component Additions:
- Add a replace(current, new) method to the component type. This is initially to allow replacement of engines within models so that multiple runs can be completed with varied models
- Add a select(filter) method to components to return a set of all child components that match the filter function
- Add a select_parent(filter) method to components to return a list of components that are the parents of items that match a filter function. The data should be returned as a set of tuples of the form (parent, <matching> child)
- Add ability to calculate drag with non-axial flows

Rocket Engine Component Additions:
- Add an equality check to engine components.
- Improve the EngineRandomize object in the engines module and characterize it to check that the resultant engines work as expected.
- Add engine canting to calculate non-axial thrust

Simulator Engine Additions
- Add actions and triggers to the simulation engine. The triggers are for engines/pyros but also to allow changing of model configurations such as staging, fall-away boosters, etc.
- Add multiple integrator types to the simulator engine. Initially Euler, Adaptive Heun (2/3), and RK4/5 should be included
- Enable configuration of integrators for minimum/maximum time step, relative tolerance, etc.
- Add callbacks to handle updated values etc. These can be used as sensor processors, etc.
- Add simulator log analysis functionality that allows for transformations of the log values and collection into plottable arrays
- Add simulation logs to keep track of general processing statistics such as number of function runs (for the iterator), number of errors, etc.

Plotting Interface
- Create a plotting interface to correctly plot multiple model logs on a single figure with simple style control
- Basic plot types include: position vs time, dynamics (acceleration/velocity/position) vs time, forces vs time, etc.

Expanded Functionality
- Add a materials library and directory (materials.py)
- Add general aerodynamic functions (aero.py)

Interfacing to other systems (long term):
- Add a RockSim importer
- Add an OpenRocket importer