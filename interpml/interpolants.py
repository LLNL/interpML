""" Interpolation techniques and bounds.

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

Additional detail provided in function docstrings.

"""

import cvxpy as cp
import numpy as np
from scipy import spatial
from scipy import special
from scipy import linalg

import time
import warnings


def delaunayError(train_x, train_y, test_x,
                  only_in_hull=False, rescale_x=True, rescale_y=True,
                  true_bound=False):
    """ Estimate the local interpolation error.

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row is the
            response value(s) for the corresponding point in train_x

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

        only_in_hull (bool, optional default=False): When set to True, only
            calculates bound for points in the convex hull and returns -1
            for all extrapolation points

        rescale_x (bool, optional default=True): When set to True, rescales
            all x values to the range [-1, 1] based on the ranges observed
            in train_x

        rescale_y (bool, optional default=True): When set to True, rescales
            all y values to the range [-1, 1] based on the ranges observed
            in train_y

        true_bound (bool, optional default=False): When set to True, reverts
            to using the worst-case bound proposed by Lux et al. (2021).

    Returns:
        np.ndarray: The predicted interpolation errors at each q in test_x

    """

    # If only one test point, reshape to (1,*)
    if len(test_x.shape) == 1:
        test_x = test_x.reshape((1,-1))
    # Rescale if needed
    if rescale_x:
        rescaled_set_x, q = __rescale_x(train_x, test_x)
    else:
        rescaled_set_x = train_x.copy()
        q = test_x.copy()
    if rescale_y:
        rescaled_set_y, yshift, yscale = __rescale_y(train_y)
    else:
        rescaled_set_y = train_y.copy()
        yshift = 0
        yscale = 1
    # Calculate constants
    d = train_x.shape[1]
    # Interpolate with DelaunaySparse
    inds = 0
    wts = 0
    res = 0
    inds, wts, res = __delaunay_simplex(rescaled_set_x, q)
    # Initialize output array
    local_errs = []
    # Loop over all outputs
    for i, qi in enumerate(q):
        # Only compute estimates for points in the convex hull?
        if not only_in_hull or res[i] < 1.0e-8:
            # If making predictions out-of-hull, then we need the projection
            if res[i] > 0:
                z_hat = __project(rescaled_set_x[inds[i]], qi)
            else:
                z_hat = qi.copy()
            # Calculate singular value of simplex i
            Ai = np.append(rescaled_set_x[inds[i]], np.ones((d+1, 1)), axis=1)
            if true_bound:
                sigmai = linalg.svd(Ai, compute_uv=False)[-1]
            else:
                sigmai = np.mean(linalg.svd(Ai, compute_uv=False))
                #sigmai = linalg.svd(Ai, compute_uv=False)[-1] ** 0.25
            # Get the min distance to nearest vertex
            minD = np.infty
            min_ind = -1
            for j, xj in zip(inds[i, :], rescaled_set_x[inds[i, :]]):
                # Check distance to z_hat for each vertex
                tempD = np.linalg.norm(z_hat - xj)
                if tempD < minD:
                    minD = tempD
                    min_ind = j
            # Get the Lipschitz constant of function and gradient
            maxL2 = __estimateLipGrad(rescaled_set_x[inds[i]],
                                      rescaled_set_y[inds[i]])
            maxL1 = __estimateLip(rescaled_set_x[inds[i]],
                                  rescaled_set_y[inds[i]])
            # Calculate max value of k^2 for chosen min_ind
            maxD2 = 0.0
            for j in inds[i, :]:
                if j != min_ind:
                    # Calculate SOS distance between verts
                    tempD = np.linalg.norm(rescaled_set_x[min_ind] -
                                           rescaled_set_x[j])
                    if true_bound:
                        maxD2 = max(tempD**2, maxD2)
                    else:
                        maxD2 += tempD ** 2
            if not true_bound:
                maxD2 /= d
            # Use the estimates to calculate ei
            ei = (maxL2 * minD ** 2 / 2
                  + np.sqrt(d) * maxL2 * maxD2 * minD / (2 * sigmai))
            ei += (res[i] * maxL1) # Add extrapolation error
            local_errs.append(ei * yscale)
        # Append -1 for out-of-hull predictions, if only_in_hull is set
        else:
            if len(rescaled_set_y.shape) < 1:
                local_errs.append(-np.ones(1))
            else:
                local_errs.append(-np.ones(rescaled_set_y.shape[1]))
    return np.asarray(local_errs)


def delaunayInterp(train_x, train_y, test_x,
                   only_in_hull=False, rescale_x=True,
                   rescale_y=True, return_weights=False):
    """ Check the value of the Delaunay interpolant on test_x

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row is the
            response value(s) for the corresponding point in train_x

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

        only_in_hull (bool, optional default=False): When set to True, only
            calculates bound for points in the convex hull and returns -1
            for all extrapolation points

        rescale_x (bool, optional default=True): When set to True, rescales
            all x values to the range [-1, 1] based on the ranges observed
            in train_x

        rescale_y (bool, optional default=True): When set to True, rescales
            all y values to the range [-1, 1] based on the ranges observed
            in train_y

    Returns:
        np.ndarray: A 1d (or 2d) array of predictions at test points in
        test_x

    """

    # If only one test point, reshape to (1,*)
    if len(test_x.shape) == 1:
        test_x = test_x.reshape((1,-1))
    # Rescale if needed
    if rescale_x:
        rescaled_set_x, q = __rescale_x(train_x, test_x)
    else:
        rescaled_set_x = train_x.copy()
        q = test_x.copy()
    if rescale_y:
        rescaled_set_y, yshift, yscale = __rescale_y(train_y)
    else:
        rescaled_set_y = train_y.copy()
        yshift = 0
        yscale = 1
    # Calculate constants
    d = train_x.shape[1]
    # Interpolate with DelaunaySparse
    inds = 0
    wts = 0
    res = 0
    inds, wts, res = __delaunay_simplex(rescaled_set_x, q.copy())
    vals = rescaled_set_y[inds]
    y_hat = np.asarray([np.dot(wts[i, :], vals[i, :])
                        for i in range(wts.shape[0])])
    # Over-extrapolation case, should never happen with current settings
    if np.all((np.array(inds[:]) == 0)):
        print("Got extrap; inds has shape", inds.shape)
    # Calculate and return array of predictions
    results = []
    for i, yi in enumerate(y_hat):
        if only_in_hull and res[i] > 1.0e-8:
            if len(rescaled_set_y.shape) < 1:
                results.append(-np.ones(1))
            else:
                results.append(-np.ones(rescaled_set_y.shape[1]))
        else:
            results.append(yi * yscale + yshift)
    if return_weights:
        return np.asarray(results), inds, wts
    else:
        return np.asarray(results)


def gpError(train_x, train_y, test_x,
            only_in_hull=False, rescale_x=True, rescale_y=True):
    """ Estimate the 95% UCB using sklearn's GP interpolant on test_x

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row is the
            response value(s) for the corresponding point in train_x

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

        only_in_hull (bool, optional default=False): When set to True, only
            calculates bound for points in the convex hull and returns -1
            for all extrapolation points

        rescale_x (bool, optional default=True): When set to True, rescales
            all x values to the range [-1, 1] based on the ranges observed
            in train_x

        rescale_y (bool, optional default=True): When set to True, rescales
            all y values to the range [-1, 1] based on the ranges observed
            in train_y

    Returns:
        np.ndarray: An array of 95% UCB distances for all q in test_x

    """

    from sklearn.gaussian_process import GaussianProcessRegressor

    # If only one test point, reshape to (1,*)
    if len(test_x.shape) == 1:
        test_x = test_x.reshape((1,-1))
    # Rescale if needed
    if rescale_x:
        rescaled_set_x, q = __rescale_x(train_x, test_x)
    else:
        rescaled_set_x = train_x.copy()
        q = test_x.copy()
    if rescale_y:
        rescaled_set_y, yshift, yscale = __rescale_y(train_y)
    else:
        rescaled_set_y = train_y.copy()
        yshift = 0
        yscale = 1
    # See if we are only using points in the convex hull
    if only_in_hull:
        interp_pts = checkInHull(rescaled_set_x, q)
    else:
        interp_pts = [i for i in range(len(q))]
    # Fit a new GP object from sklearn
    gp = GaussianProcessRegressor().fit(rescaled_set_x, rescaled_set_y)
    # Loop over all test points and populate error array
    local_errs = []
    for i, qi in enumerate(q):
        if i in interp_pts:
            # Get the std dev at qi, 2*sd is the 95% confidence bound
            mean, sd = gp.predict([qi], return_std=True)
            local_errs.append(2*sd.flatten() * yscale)
        else:
            if len(rescaled_set_y.shape) < 1:
                local_errs.append(-np.ones(1))
            else:
                local_errs.append(-np.ones(rescaled_set_y.shape[1]))
    return np.asarray(local_errs)


def gpInterp(train_x, train_y, test_x,
             only_in_hull=False, rescale_x=True, rescale_y=True):
    """ Check the value of scikit-learn's Gaussian Proc interpolant on test_x

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row is the
            response value(s) for the corresponding point in train_x

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

        only_in_hull (bool, optional default=False): When set to True, only
            calculates bound for points in the convex hull and returns -1
            for all extrapolation points

        rescale_x (bool, optional default=True): When set to True, rescales
            all x values to the range [-1, 1] based on the ranges observed
            in train_x

        rescale_y (bool, optional default=True): When set to True, rescales
            all y values to the range [-1, 1] based on the ranges observed
            in train_y

    Returns:
        np.ndarray: A 1d (or 2d) array of predictions at test points in
        test_x

    """

    from sklearn.gaussian_process import GaussianProcessRegressor

    # If only one test point, reshape to (1,*)
    if len(test_x.shape) == 1:
        test_x = test_x.reshape((1,-1))
    # Rescale if needed
    if rescale_x:
        rescaled_set_x, q = __rescale_x(train_x, test_x)
    else:
        rescaled_set_x = train_x.copy()
        q = test_x.copy()
    if rescale_y:
        rescaled_set_y, yshift, yscale = __rescale_y(train_y)
    else:
        rescaled_set_y = train_y.copy()
        yshift = 0
        yscale = 1
    # See if we are only using points in the convex hull
    if only_in_hull:
        interp_pts = checkInHull(rescaled_set_x, q)
    else:
        interp_pts = [i for i in range(len(q))]
    # Fit a new GP object from sklearn
    gp = GaussianProcessRegressor().fit(rescaled_set_x, rescaled_set_y)
    # Collect all predictions
    results = []
    for i, qi in enumerate(q):
        if i in interp_pts:
            results.append(gp.predict([qi])[0] * yscale + yshift)
        else:
            if len(rescaled_set_y.shape) < 1:
                results.append(-np.ones(1))
            else:
                results.append(-np.ones(rescaled_set_y.shape[1]))
    return np.asarray(results)


def rbfError(train_x, train_y, test_x,
             only_in_hull=False, rescale_x=True, rescale_y=True):
    """ Estimate the interpolation error of a thin-plate spline RBF

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row is the
            response value(s) for the corresponding point in train_x

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

        only_in_hull (bool, optional default=False): When set to True, only
            calculates bound for points in the convex hull and returns -1
            for all extrapolation points

        rescale_x (bool, optional default=True): When set to True, rescales
            all x values to the range [-1, 1] based on the ranges observed
            in train_x

        rescale_y (bool, optional default=True): When set to True, rescales
            all y values to the range [-1, 1] based on the ranges observed
            in train_y

    Returns:
        np.ndarray: An array of error estimates

    """

    # If only one test point, reshape to (1,*)
    if len(test_x.shape) == 1:
        test_x = test_x.reshape((1,-1))
    # Rescale if needed
    if rescale_x:
        rescaled_set_x, q = __rescale_x(train_x, test_x)
    else:
        rescaled_set_x = train_x.copy()
        q = test_x.copy()
    if rescale_y:
        rescaled_set_y, yshift, yscale = __rescale_y(train_y)
    else:
        rescaled_set_y = train_y.copy()
        yshift = 0
        yscale = 1
    # See if we are only using points in the convex hull
    if only_in_hull:
        interp_pts = checkInHull(rescaled_set_x, q)
    else:
        interp_pts = [i for i in range(len(q))]
    # Get constants
    L = __estimateLip(rescaled_set_x, rescaled_set_y)
    # Now loop over all test points and try to estimate the bounds
    local_errs = []
    for i, qi in enumerate(q):
        if i in interp_pts:
            # h is the distance to nearest neighbor
            hi = np.min(spatial.distance.cdist(rescaled_set_x, [qi]))
            # Estimated norm of F in Phi's RKHS by Lip. constant
            Fxi = L
            # Estimate P(x)
            if hi > 0.0 and hi < 1.0:
                Pxi = hi * np.sqrt(np.log(1 / hi))
            else:
                Pxi = hi
            # Use theorem from paper
            local_errs.append(Fxi * Pxi * yscale)
        else:
            local_errs.append(-1)
    return np.asarray(local_errs)


def rbfInterp(train_x, train_y, test_x,
              only_in_hull=False, rescale_x=True, rescale_y=True):
    """ Check the value of scipy's RBF interpolant on test_x

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row is the
            response value(s) for the corresponding point in train_x

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

        only_in_hull (bool, optional default=False): When set to True, only
            calculates bound for points in the convex hull and returns -1
            for all extrapolation points

        rescale_x (bool, optional default=True): When set to True, rescales
            all x values to the range [-1, 1] based on the ranges observed
            in train_x

        rescale_y (bool, optional default=True): When set to True, rescales
            all y values to the range [-1, 1] based on the ranges observed
            in train_y

    Returns:
        np.ndarray: A 1d (or 2d) array of predictions at test points in
        test_x

    """

    from scipy.interpolate import RBFInterpolator

    # If only one test point, reshape to (1,*)
    if len(test_x.shape) == 1:
        test_x = test_x.reshape((1,-1))
    # Rescale if needed
    if rescale_x:
        rescaled_set_x, q = __rescale_x(train_x, test_x)
    else:
        rescaled_set_x = train_x.copy()
        q = test_x.copy()
    if rescale_y:
        rescaled_set_y, yshift, yscale = __rescale_y(train_y)
    else:
        rescaled_set_y = train_y.copy()
        yshift = 0
        yscale = 1
    # See if we are only using points in the convex hull
    if only_in_hull:
        interp_pts = checkInHull(rescaled_set_x, q)
    else:
        interp_pts = [i for i in range(len(q))]
    # Fit model
    rbf = RBFInterpolator(rescaled_set_x, rescaled_set_y)
    # Collect predictions
    results = []
    for i, qi in enumerate(q):
        if i in interp_pts:
            results.append(rbf([qi])[0] * yscale + yshift)
        else:
            results.append(-1)
    return np.asarray(results)


def reluMlpInterp(train_x, train_y, test_x,
                  only_in_hull=False, rescale_x=True, rescale_y=True,
                  use_valid=True, nrestarts=100, early_terminate=50):
    """ Check the value of scikit-learn's ReLU MLP regressor on test_x

    WARNING: When run with default settings, this model could take a long
    time...

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row is the
            response value(s) for the corresponding point in train_x

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

        only_in_hull (bool, optional default=False): When set to True, only
            calculates bound for points in the convex hull and returns -1
            for all extrapolation points

        rescale_x (bool, optional default=True): When set to True, rescales
            all x values to the range [-1, 1] based on the ranges observed
            in train_x

        rescale_y (bool, optional default=True): When set to True, rescales
            all y values to the range [-1, 1] based on the ranges observed
            in train_y

        use_valid (bool, optional default=True): When set to True,
            the neural network model is retrained multiple times with
            different validation sets, and the model that achieves the
            best validation score is used

        nrestarts (int, optional default=100): Only used when
            use_valid=True. The neural network model will be
            retrained this many times and the "best" model will be
            used for making predictions

        early_terminate (int, optional default=50): Only used when
            use_valid=True. If this many iterations pass without
            observing any improvement to the validation score, then
            the validation loop will exit early

    Returns:
        np.ndarray: A 1d (or 2d) array of predictions at test points in
        test_x

    """

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor

    # If only one test point, reshape to (1,*)
    if len(test_x.shape) == 1:
        test_x = test_x.reshape((1,-1))
    # Rescale if needed
    if rescale_x:
        rescaled_set_x, q = __rescale_x(train_x, test_x)
    else:
        rescaled_set_x = train_x.copy()
        q = test_x.copy()
    if rescale_y:
        rescaled_set_y, yshift, yscale = __rescale_y(train_y)
    else:
        rescaled_set_y = train_y.copy()
        yshift = 0
        yscale = 1
    if len(rescaled_set_y.shape) == 2 and rescaled_set_y.shape[1] == 1:
        rescaled_set_y = rescaled_set_y.flatten()
        expand = True
    else:
        expand = False
    # See if we are only using points in the convex hull
    if only_in_hull:
        interp_pts = checkInHull(rescaled_set_x, q)
    else:
        interp_pts = [i for i in range(len(q))]
    # Use a randomized validation set and perform randomized restarts
    best_mlp = None
    best_valid = None
    stop_iter = 0
    for i in range(nrestarts):
        # Create a new ReLU MLP from sklearn
        mlp = MLPRegressor(hidden_layer_sizes=(100,100,100), # 3 hidden layers
                           activation="relu", # ReLU activ on hidden layers
                           solver="adam", # minimize with Adam
                           alpha=1.0e-8, # regularization constant
                           early_stopping=use_valid, # toggle validation set
                           # Validation set hyperparams
                           tol=1.0e-6, # Early-stopping criteria
                           validation_fraction=0.1, # % data for validation
                           # Adam hyperparams
                           batch_size=min(20, # 20 data points per batch
                                          int(rescaled_set_x.shape[0] * 0.9)),
                           learning_rate_init=1.0e-3, # init learning rate
                           max_iter=200, # 200 epoch limit
                           shuffle=True # reshuffle samples in each epoch
                          )
        # Fit the MLP, but catch and ignore all convergence warnings
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                mlp.fit(rescaled_set_x, rescaled_set_y)
        except ConvergenceWarning:
            pass
        # Update the current hyperparams and continue training
        mlp.set_params(batch_size=min(200,
                                      int(rescaled_set_x.shape[0] * 0.9)),
                       max_iter=100,
                       warm_start=True
                      )
        # Resume training the MLP, ignoring convergence warnings
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                mlp.fit(rescaled_set_x, rescaled_set_y)
        except ConvergenceWarning:
            pass
        # Check the validation score for improvement
        if best_valid is None or mlp.best_validation_score_ > best_valid:
            best_mlp = mlp
            stop_iter = 0
        else:
            stop_iter += 1
        if not use_valid:
            break # Break immediately if no validation set
        elif stop_iter >= early_terminate:
            break # Break after so many iterations with no improvement
    # Collect results
    results = []
    for i, qi in enumerate(q):
        if i in interp_pts:
            if np.issubdtype(type(yscale), np.floating):
                yscale=np.array(yscale)
            results.append(best_mlp.predict([qi]).flatten().tolist() * yscale
                           + yshift)
        else:
            results.append(-1)
    results = np.asarray(results)
    return results


### --------- Private helper functions --------- ###


def checkInHull(train_x, test_x, rescale_x=False):
    """ Check percentage of a test set in the convex hull of training data.

    Solves the LP formulation in Section 2.1 of ACM TOMS Alg. 1012 and
    checks whether solution is unbounded (e.g., if the dual is infeasible).

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

        rescale_x (bool, optional default=True): When set to True, rescales
            all x values to the range [-1, 1] based on the ranges observed
            in train_x

    Returns:
        float: The proportion (between 0 and 1) of points in test_x that
        are inside the convex hull of train_x

    """

    # Rescale if needed
    if rescale_x:
        rescaled_train_x, rescaled_test_x = __rescale_x(train_x, test_x)
    else:
        rescaled_train_x = train_x.copy()
        rescaled_test_x = test_x.copy()
    # Track percentage of points
    total_in_hull = []
    total_holdout = rescaled_test_x.shape[0]
    # Allocate memory for problem
    A = np.ones((rescaled_train_x.shape[0], rescaled_train_x.shape[1] + 1))
    b = np.ones(rescaled_train_x.shape[0])
    c = np.ones(rescaled_test_x.shape[1] + 1)
    # Create static matrices/vectors (A and b)
    for i, xi in enumerate(rescaled_train_x):
        A[i, :-1] = -xi[:]
        A[i, -1] = 1.0
        b[i] = np.linalg.norm(xi) ** 2
    # Adjust c for each solve
    for i, qi in enumerate(rescaled_test_x):
        c[:-1] = -qi[:]
        # Solve the dual formulation
        y = cp.Variable(b.size)
        prob = cp.Problem(cp.Minimize(b.T @ y), [A.T @ y == c, y >= 0])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                prob.solve(max_iters=np.prod(rescaled_train_x.shape)+50,
                           # verbose=True,  # Uncomment for DEBUG
                           solver="ECOS"
                )
                # print(prob.status) # Debug statement, uncomment for status
                # Count feasible solutions (i.e., unbounded primal problem)
                if prob.status != 'infeasible':
                    total_in_hull.append(i)
        except: # Catch all errors and warnings
            q_hat = __project(rescaled_train_x, qi)
            if np.linalg.norm(q_hat - qi) < 1.0e-8:
                total_in_hull.append(i)
    return total_in_hull


def __delaunay_simplex(pts, q, allow_extrapolation=True, print_errors=True,
                       parallel=False, pmode=None, chain=None,
                       ibudget=10000, epsilon=2**(-23), check_spacing=False):
    """Find the Delaunay simplices containing test pts using DelaunaySparse.

    Helper function for delaunayInterpolate() and other simplex-based
    estimators.

    Warning: this function requires that
    ``DelaunaySparse/python/delsparse.py`` be contained in
    this directory, and the Fortran dependencies be installed in a
    subdirectory. Upon the initial import, the dependencies will be built
    with gfortran.
    
    Args:
      pts -- 2D Numpy array of float64 points, where each row is one point.
      q -- 2D numpy array of float64 points where Delaunay predictions
           are to be made, where each row is one point.
    
    Returns:
      (indices, weights, residuals) -- Where "indices" is a 2D NumPy array
         of integers and each row, i, enumerates the indices of rows in
         "pts" that are the vertices of the simplex containing q[i],
         and each corresponding row of weights (a 2D NumPy array of float64)
         provides the convex weights such that q[i] = np.dot(pts[indices[i]],
         weights[i]). Each entry residuals[i] contains the projection
         residual if q[i] was an extrapolation point, otherwise it is 0.

    """

    # Make sure delsparse.py and its fortran dependencies are in this dir,
    # and gfortran is installed
    from .DelaunaySparse.python import delsparse
 
    # Enable parallelism.
    if parallel:
        if pmode is None:
            if len(q.shape) == 1 or q.shape[0] == 1:
                lpmode = 2
            else:
                lpmode = 1
        else:
            lpmode = pmode
        if lpmode == 3:
            import os
            os.environ["OMP_NESTED"] = "TRUE"
    # Get the predictions from DelaunaySparse
    pts_in = np.asarray(pts.T, dtype=np.float64, order="F")
    p_in = np.asarray(q.T, dtype=np.float64, order="F")
    simp_out = np.ones(shape=(p_in.shape[0]+1, p_in.shape[1]), 
                       dtype=np.int32, order="F")
    weights_out = np.ones(shape=(p_in.shape[0]+1, p_in.shape[1]), 
                          dtype=np.float64, order="F")
    error_out = np.ones(shape=(p_in.shape[1],), 
                        dtype=np.int32, order="F")
    residuals_out = np.ones(shape=(p_in.shape[1],), 
                            dtype=np.float64, order="F")
    if parallel:
        delsparse.delaunaysparsep(pts_in.shape[0], pts_in.shape[1],
                                  pts_in, p_in.shape[1], p_in, simp_out,
                                  weights_out, error_out, extrap=1.0e8,
                                  pmode=lpmode, ibudget=ibudget,
                                  eps=epsilon, chain=chain,
                                  exact=check_spacing, rnorm=residuals_out)
    else:
        delsparse.delaunaysparses(pts_in.shape[0], pts_in.shape[1],
                                  pts_in, p_in.shape[1], p_in, simp_out,
                                  weights_out, error_out, extrap=1.0e8, 
                                  ibudget=ibudget, eps=epsilon,
                                  chain=chain, exact=check_spacing,
                                  rnorm=residuals_out)
    # Remove "extrapolation" errors if the user doesn't care.
    if allow_extrapolation:
        error_out = np.where(error_out == 1, 0, error_out)
    else:
        if 1 in error_out:
            class Extrapolation(Exception): pass
            raise(Extrapolation("Encountered extrapolation point when " +
                                "making Delaunay prediction."))
    # Handle any errors that may have occurred.
    if (sum(error_out) != 0):
        if print_errors:
            unique_errors = sorted(np.unique(error_out))
            print(" [Delaunay errors:",end="")
            for e in unique_errors:
                if (e == 0): continue
                indices = tuple(str(i) for i in range(len(error_out))
                                if (error_out[i] == e))
                if (len(indices) > 5): indices = (indices[:2] + ('...',) +
                                                  indices[-2:])
                print(" %3i"%e,"at","{"+",".join(indices)+"}", end=";")
            print("] ")
        # Reset the errors to simplex of 1s (to be 0) and weights of 0s.
        bad_indices = (error_out > (1 if allow_extrapolation else 0))
        simp_out[:,bad_indices] = 1
        weights_out[:,bad_indices] = 0
    # Adjust the output simplices and weights to be expected shape.
    indices  = simp_out.T - 1
    weights = weights_out.T
    residuals = residuals_out
    # Return the appropriate shaped pair of points and weights
    return (indices, weights, residuals)


def __estimateLip(train_x, train_y):
    """ Check the max Lipschitz constaint observed in training data.

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row
            is a response value for the corresponding point in train_x

    Returns:
        float: The max Lipschitz constant observed across all training data

    """

    result = []
    if len(train_y.shape) == 2:
        dims = train_y.shape[1]
    else:
        dims = 1
    for dim in range(dims):
        pDist = spatial.distance.pdist(train_x)
        for i in range(train_x.shape[0]):
            for j in range(train_x.shape[0]):
                if j > i:
                    ind = train_x.shape[0] * i + j - ((i + 2) * (i + 1)) // 2
                    if dims > 1:
                        pDist[ind] = (np.abs(train_y[i, dim] -
                                             train_y[j, dim]) / pDist[ind])
                    else:
                        pDist[ind] = (np.abs(train_y[i] -
                                             train_y[j]) / pDist[ind])
        result.append(np.max(pDist))
    if len(result) == 1:
        return result[0]
    else:
        return np.asarray(result)


def __estimateLipGrad(train_x, train_y):
    """ Estimate the max Lipschitz constant of gradient via 3pt DD

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        train_y (numpy.ndarray): A 1d or 2d float array, where each row
            contains response value(s) for the corresponding point in train_x

    Returns:
        float or numpy.ndrray: The max Lipschitz constant of gradient
        observed across all training data based on 3pt divided difference
        estimates

    """

    # Determine whether to reshape training data
    reshape_y = False
    if len(train_y.shape) == 1:
        reshape_y = True
        train_y2 = train_y.reshape((train_y.size, 1))
    else:
        train_y2 = train_y.copy()
    # Initialize counters
    maxLipGrad = np.zeros(train_y2.shape[1])
    tempLipGrad = np.zeros(train_y2.shape[1])
    # Loop over all triples
    for i in range(train_x.shape[0]):
        for j in range(i+1, train_x.shape[0]):
            for k in range(j+1, train_x.shape[0]):
                # All ways to compute 3pt divided difference
                xi = train_x[i]
                Lij = train_y2[i, :] - train_y2[j, :]
                xij = train_x[i] - train_x[j]
                xj = train_x[j]
                Lik = train_y2[i, :] - train_y2[k, :]
                xik = train_x[i] - train_x[k]
                xk = train_x[k]
                Ljk = train_y2[j, :] - train_y2[k, :]
                xjk = train_x[j] - train_x[k]
                for ii in range(train_y2.shape[1]):
                    tempLipGrad[ii] = (np.linalg.norm(Lij[ii] / xij
                                                      - Lik[ii] / xik) /
                                       np.linalg.norm(xjk))
                maxLipGrad = np.max(np.asarray([tempLipGrad.tolist(),
                                                maxLipGrad.tolist()]), axis=0)
                for ii in range(train_y2.shape[1]):
                    tempLipGrad[ii] = (np.linalg.norm(Lij[ii] / xij
                                                      - Ljk[ii] / xjk) /
                                       np.linalg.norm(xik))
                maxLipGrad = np.max(np.asarray([tempLipGrad.tolist(),
                                                maxLipGrad.tolist()]), axis=0)
                for ii in range(train_y2.shape[1]):
                    tempLipGrad[ii] = (np.linalg.norm(Lik[ii] / xik
                                                      - Ljk[ii] / xjk) /
                                       np.linalg.norm(xij))
                maxLipGrad = np.max(np.asarray([tempLipGrad.tolist(),
                                                maxLipGrad.tolist()]), axis=0)
    # Reshape outputs if needed
    if reshape_y:
        maxLipGrad_out = maxLipGrad[0]
    else:
        maxLipGrad_out = maxLipGrad
    return maxLipGrad_out


def __print_time(old_time_stamp):
    """ For timing. """

    new_time_stamp = time.perf_counter()
    # Comment the line below to turn off time printing
    print(f"\t\t\t\t\t\t time: {np.round(new_time_stamp - old_time_stamp,5)}")
    return new_time_stamp


def __project(train, z):
    """ Project x onto the convex hull of train.

    Solves the QP formulation in Section 4.2 of ACM TOMS Alg. 1012.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        z (numpy.ndarray): A 1d float array, containing the point to project

    Returns:
        numpy.ndarray: A 1d array containing the result of the projection

    """

    # Allocate memory for problem
    A = train.copy()
    b = z.copy()
    # Solve the QP
    x = cp.Variable(A.shape[0])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A.T @ x - b)),
                      [cp.sum(x) == 1, x >= 0])
    prob.solve(max_iters=np.prod(train.shape)+50,
               # verbose=True,
               solver="ECOS")
    z_hat = np.dot(x.value, train).flatten()
    return z_hat


def __rescale_x(train_x, test_x):
    """ Rescale training and testing points to improve conditioning.

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        test_x (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

    Returns:
        rescale_train (numpy.ndarray), rescale_test (numpy.ndarray):
        The two rescaled datasets, rescaled dimension-wise to fill
        the cube [-1, 1]

    """

    __shift__ = np.zeros(train_x.shape[1])
    __scale__ = np.ones(train_x.shape[1])
    __shift__[:] = np.min(train_x, axis=0)
    __scale__[:] = np.max(np.array([xi - __shift__[:]
                                    for xi in train_x]), axis=0)
    __scale__[np.where(__scale__ < 1.0e-8)] = 2.0
    rescaled_train = np.array([(xi - __shift__) / __scale__
                               for xi in train_x]) * 2 - 1
    rescaled_test = np.array([(xi - __shift__) / __scale__
                              for xi in test_x]) * 2 - 1
    return rescaled_train, rescaled_test


def __rescale_y(train_y):
    """ Rescale training obvervations to improve conditioning.

    Args:
        train_y (numpy.ndarray): A 1d or 2d float array, where each row
            contains response value(s) for the corresponding point in train_x

    Returns:
        rescale_train (numpy.ndarray), yshift (numpy.ndarray),
        yscale (numpy.ndarray): The rescaled training outputs
        (to the cube [-1, 1]) and the shift and scale factors

    """

    # Calculate the shift and scale factors for 2 different cases
    if len(train_y.shape) > 1:
        yshift = np.zeros(train_y.shape[1])
        yscale = np.ones(train_y.shape[1])
        yshift[:] = np.min(train_y, axis=0)
        yscale[:] = np.max(np.array([yi - yshift[:]
                                     for yi in train_y]), axis=0)
        yscale[np.where(yscale < 1.0e-8)] = 2.0
    else:
        yshift = np.min(train_y)
        yscale = np.max(np.array([yi - yshift for yi in train_y]))
        # if yscale < 1.0e-8:
        #     yscale = 2.0
    
    yshift = yshift + (yscale / 2)
    yscale = yscale / 2
    rescaled_train = np.array([(yi - yshift) / yscale
                               for yi in train_y])
    return rescaled_train, yshift, yscale


if __name__ == '__main__':
    """ Driver code for debugging and testing """

    train_x = np.random.random_sample((100, 4)) * 2 - 1
    train_y = np.sum(0.5 * train_x**2 + 3.0 * train_x, axis=1)
    train_y = np.vstack((np.sum(train_x**2+train_x, axis=1).flatten(),
                         np.sum(np.sin(train_x), axis=1).flatten())).T
    test_x = np.random.random_sample((4,4)) * 2 -1
    test_y = np.sum(0.5 * test_x**2 + 2.0 * test_x, axis=1)
    test_y = np.vstack((np.sum(test_x**2+test_x, axis=1).flatten(),
                        np.sum(np.sin(test_x), axis=1).flatten())).T
    ots = time.perf_counter() # ots = old time stamp
    # Test helpers
    print(f"In convex hull?: \t{checkInHull(train_x, test_x)}") 
    ots = __print_time(ots)
    ## Following 2 are deceptively long b/c you don't call on all data usually
    #print(f"Estimated Lipschitz: \t{__estimateLip(train_x, train_y)}")
    #ots = __print_time(ots)
    #print(f"Estimated Lip Grad: \t{__estimateLipGrad(train_x, train_y)}")
    #ots = __print_time(ots)
    # Test Delaunay
    exp_errors = delaunayError(train_x, train_y, test_x)
    print(f"Exp Delaunay Error:    \t{exp_errors}")
    ots = __print_time(ots)
    errors = np.abs(delaunayInterp(train_x, train_y, test_x) - test_y)
    print(f"Delaunay interp Error: \t{errors}")
    ots = __print_time(ots)
    # Test GP
    exp_errors = gpError(train_x, train_y, test_x)
    print(f"Exp GP Error:    \t{exp_errors}")
    ots = __print_time(ots)
    errors = np.abs(gpInterp(train_x, train_y, test_x) - test_y)
    print(f"GP interp Error: \t{errors}")
    ots = __print_time(ots)
    # Test RBF
    exp_errors = rbfError(train_x, train_y, test_x)
    print(f"Exp RBF Error:    \t{exp_errors}")
    ots = __print_time(ots)
    errors = np.abs(rbfInterp(train_x, train_y, test_x) - test_y)
    print(f"RBF interp Error: \t{errors}")
    ots = __print_time(ots)
    # Test ReLU net
    errors = np.abs(reluMlpInterp(train_x, train_y, test_x,
                                  nrestarts=50, early_terminate=20) - test_y)
    print(f"ReLU MLP Error:   \t{errors}")
    ots = __print_time(ots)

