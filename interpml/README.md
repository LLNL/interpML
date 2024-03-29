# Interpolation Methods

This directory contains implementations for the interpolation methods used
in our paper.

Interfaces to all methods are contained in the ``interpolants.py`` file,
but the header is duplicated below to summarize usage.

Contains the following public functions:
 - ``{method}Error(train_x, train_y, test_x,
     only_in_hull=False, rescale_x, rescale_y)``
     and
 - ``{method}Interp(train_x, train_y, test_x,
     only_in_hull=False, rescale_x, rescale_y)``
where ``method`` specifies one of the following:
 - ``delaunay`` (Delaunay interpolation);
 - ``rbf`` (thin-plate spline RBF);
 - ``gp`` (Gaussian process interpolation / kriging).

The required arguments are as follows:
 - ``train_x`` (2d numpy.ndarray) contains the training features /
    interpolation nodes;
 - ``train_y`` (1d or 2d numpy.ndarray) contains the training values /
   known response values corresponding to points in ``train_x``;
 - ``test_x`` (2d numpy.ndarray) contains the test points to evaluate
   after training / fitting a model based on the values in ``train_x``
   and ``train_y``.

The following values are returned:
 - ``test_y`` (1d or 2d numpy.ndarray) contains the predicted response
   values corresponding to each point in ``test_x``.

The optional arguments are as follows:
 - ``only_in_hull`` causes only interpolation points to be evaluated, when set
   to ``True``;
 - ``rescale_x`` causes ``x`` inputs (features) to be rescaled on input;
 - ``rescale_y`` causes ``y`` inputs (predictions) to be rescaled on input and
   de-scaled on output.

The Delaunay methods have additional optional inputs:
 - ``delaunayError`` accepts an additional optional input ``true_bound``
   (defaults to ``False``) which can be set to ``True`` to use a true
   worst-case bound that is often overly pessimistic in practice;
 - ``delaunayInterp`` accepts an additional optional input ``return_weights``
   (defaults to ``False``) which can be set to ``True`` to obtain the
   corresponding training points and weights used to make the prediction.
   When set to ``True``, the indices of the training points and corresponding
   weights will be returned as additional outputs.

Additionally, although no "Error" function is provided, the following function
trains (several) ReLU MLPs on the training set, selects the best network via a
held-out validation set, then evaluates its performance on the test set:
 - ``reluMlpInterp(train_x, train_y, test_x,
                   only_in_hull=False, rescale_x=True, rescale_y=True,
                   use_valid=True, nrestarts=100, early_terminate=50)``

For ``reluMlpInterp``, the following optional arguments can be used to adjust
training hyperparameters. The default values are recommended for most SciML
regression problems based on our experience, but may take a significant amount
of time to evaluate (therefore, could be lowered to save compute time):
 - ``use_valid`` (default ``True``) will train a single ReLU MLP on the full
   training set and use it when set to ``False``, otherwise holds out ``10%``
   for validation;
 - ``nrestarts`` (default 100) maximum number of ReLU MLPs to retrain during
   the validation phase;
 - ``early_terminate`` (default 50) if no significant improvement is shown
   on the validation set after training this many models, the validation
   loop will terminate early.

Other helper functions are also provided (not documented above).
Notably, a data deduplicator is provided in ``dedup.py``:
 - ``deDupe(train, tol=1e-4)`` deduplicates rows in ``train[x]`` that are the
   same up to the given ``tol`` and averages the corresponding values in
   ``train[y]``.

Additional detail provided in function docstrings.

## Set-Up

The ``interpolants.py`` script has the following dependancies:
 - ``python>=3.8``
 - ``cvxpy>=1.3``
 - ``networkx>=3.1``
 - ``numpy>=1.17``
 - ``scipy>=1.10``
 - ``scikit-learn>=1.2``
 - a ``gfortran`` compiler supporting the Fortran 2003 standard.

The ``DelaunaySparse`` package is a submodule. To install, run
the following commands

```
git submodule init
git submodule update
```

From within a Python interpreter, try to import the ``interpolants`` module
and call the ``interpolants.delaunayInterp(train_x, train_y, test_x)``
function. The Fortran binaries will be built on the first call. It is normal
to see the build command and possibly a few warnings if you have an older
compiler, but there should be no build errors.

The following test is recommended to make sure it is working:
```
>>> import numpy as np
>>> import interpolants
>>> train_x = np.concatenate((np.eye(3), np.zeros((1, 3))))
>>> train_y = np.ones((4, 1))
>>> test_x = np.zeros((1, 3))
>>> interpolants.delaunayInterp(train_x, train_y, test_x)
```
The result of the above calls should be ``array([[1.]])`` (or similar).

In order to make externally usable, pip install from the parent directory.
