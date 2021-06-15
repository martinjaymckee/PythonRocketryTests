# PythonRocketryTests

This is a collection of disparate python scripts to test different rocket related concepts, formulas, and so forth. The primary goal of these scripts is to test ideas quickly and, as such, they are often incomplete and do not handle obvious corner cases in a problem.  This may result in dimensionality being reduced (i.e. 3D down to 1D), or known error sources being disregarded in simulations.  In any case, none of these tests are expected to be sufficient - in and of themselves - to sufficiently model a problem that it can be used for a final solution.

There is little documentation in any of the scripts at the moment, so reusability is low.  Eventually it would be nice to clean up the code, combine it, and add some minimal documentation (even if only directly at the top of each script) to make reuse easier.

## Dependencies
There are a number of Python libraries used in the various tests.  To run them all, the following modules must be installed.

- numpy
- numpy-quaternions (it may be better to replace this anyway)
- scipy
- pyserial
- matplotlib
- seaborn
- progress (progress bar library)
- pandas
