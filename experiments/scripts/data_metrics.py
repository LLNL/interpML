import numpy as np
import pandas as pd
import itertools
import warnings
import time


def convexHullPercent(train, holdout, return_list=False):
    """ Check percentage of a holdout set in the convex hull of training data.

    Solves the LP formulation in Section 2.1 of ACM TOMS Alg. 1012 and
    checks whether solution is unbounded (e.g., if the dual is infeasible).

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        holdout (numpy.ndarray): A 2d float array, where each row indicates a
            testing point

    Returns:
        float: The proportion (between 0 and 1) of points in holdout that
        are inside the convex hull of train

    """

    import cvxpy as cp

    # Track percentage of points
    total_in_hull = []
    total_holdout = holdout.shape[0]
    # Allocate memory for problem
    A = np.ones((train.shape[0], train.shape[1] + 1))
    b = np.ones(train.shape[0])
    c = np.ones(holdout.shape[1] + 1)
    # Create static matrices/vectors (A and b)
    for i, xi in enumerate(train):
        A[i, :-1] = -xi[:]
        A[i, -1] = 1.0
        b[i] = np.linalg.norm(xi) ** 2
    # Adjust c for each solve
    for i, qi in enumerate(holdout):
        c[:-1] = -qi[:]
        # Solve the dual formulation
        y = cp.Variable(b.size)
        prob = cp.Problem(cp.Minimize(b.T @ y), [A.T @ y == c, y >= 0])
        prob.solve()
        # print(prob.status) # Debug statement, uncomment to print status
        # Count feasible solutions (i.e., unbounded for primal problem)
        if prob.status != 'infeasible':
            total_in_hull.append(i)
    if return_list:
        return total_in_hull
    else:
        return len(total_in_hull) / total_holdout


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

    import cvxpy as cp

    # Allocate memory for problem
    A = train.copy()
    b = z.copy()
    # Solve the QP
    x = cp.Variable(A.shape[0])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A.T @ x - b)),
                      [cp.sum(x) == 1, x >= 0])
    prob.solve()
    z_hat = np.dot(x.value, train).flatten()
    return z_hat


def maximin(train):
    """ Check the maximin of the training points.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

    Returns:
        float: The min ||xi - xj|| over all xi != xj in train.

    """

    from scipy.spatial import distance

    return np.min(distance.pdist(train))


def minimax(train):
    """ Check the minimax of the training points.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

    Returns:
        float: The max (min ||x - xj|| for xj in train) over all x in X.

    """

    from scipy.spatial import distance
    from scipy.optimize import minimize

    # Get metadata and create a helper lambda
    d = train.shape[1]
    MinDist = lambda x: np.min(distance.cdist(train, [x]))
    # Start by sampling the center of the hypercube
    max_x = np.zeros(d)
    max_dist = MinDist(max_x)
    old_max_x = np.random.random_sample(d) * 2 - 1
    # Identify the maximum distance corner of the cube
    vi = np.zeros(d)
    for i in range(2**d):
        for j in range(d):
            vi[j] = ((i >> j) % 2 - 0.5) * 2
        disti = MinDist(vi)
        if disti > max_dist:
            old_max_x[:] = max_x[:] # Keep the old max
            max_dist = disti
            max_x = vi
    # Use optimization to see if any of our solutions can be improved
    res = minimize(lambda x: -MinDist(x), max_x, method="Powell",
                   bounds=[(-1, 1) for i in range(d)])
    if MinDist(res['x']) > max_dist:
        max_x = res['x']
        max_dist = MinDist(res['x'])
    res = minimize(lambda x: -MinDist(x), old_max_x, method="Powell",
                   bounds=[(-1, 1) for i in range(d)])
    if MinDist(res['x']) > max_dist:
        max_x = res['x']
        max_dist = MinDist(res['x'])
    return max_dist


def samplingDensity(train, tolerance=0.5):
    """ Check how many training points we have, normalized by volume.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

    Returns:
        float: n / 2^d.

    """

    return train.shape[0] / (2 ** train.shape[1])


def samplingDistances(train, tolerance=1):
    """ Check how many training points we have, normalized by dimension.

    Assuming a multilevel factorial design, we need at least
    2 sqrt(d)/tolerance + 1 levels to maintain a minimax distance of tolerance
    in the cube [-1, 1]^d. Then n=(2*sqrt(d)/tolerance + 1)^d total samples
    would be required to maintain the minimax distance (in 2-norm).
    To calculate the sampling density, we normalize the number of training
    points by the number of samples needed to maintain a full factorial
    design with guaranteed minimax distance.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

        tolerance (float, optional): The tolerance used to calculate the
            sampling density. Defaults to 1.

    Returns:
        float: The number of training points normalized by
        (sqrt(d)/2*tolerance)^d. If <1, we do not have enough points to
        achieve tolerance via multilevel factorial design.
        If >1, we have more than enough points to achieve a multilevel
        factorial design.

    """

    return train.shape[0] / ((2 * np.sqrt(train.shape[1]) / tolerance + 1)
                             ** train.shape[1])


def discrepancy(train):
    """ Check the centered discrepancy of the training points.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

    Returns:
        float: The centered discrepancy of the training sample.

    """

    from scipy.stats import qmc

    aug_train = np.zeros(train.shape)
    aug_train = (train + 1) / 2
    return qmc.discrepancy(aug_train)


def eOptimality(train):
    """ Check the minimum eigenvalue of the linear Fisher information matrix.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

    Returns:
        float: The minimum eigenvalue of train' train (E-optimality criterion)

    """

    from scipy import linalg

    return linalg.svd(train, compute_uv=False)[-1]


def invCondition(train):
    """ Check the inverse condition number of the linear info matrix.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

    Returns:
        float: The inverse condition number of train' train (1 = perfect,
        0 = terrible)

    """

    from scipy import linalg

    sing_vals = linalg.svd(train, compute_uv=False)
    return sing_vals[-1] / sing_vals[0]


def covarianceCondition(train):
    """ Check the inverse condition number of the covariance matrix.

    Args:
        train (numpy.ndarray): A 2d float array, where each row indicates a
            training point

    Returns:
        float: The inverse condition number of the covariance matrix
        (1 = perfect, 0 = terrible)

    """

    from scipy import linalg

    mean = np.sum(train, axis=0) / train.shape[0]
    cov = np.zeros((train.shape[1], train.shape[0]))
    for i, xi in enumerate(train):
        cov[:, i] = xi - mean
    s = linalg.svd(cov, compute_uv=False)
    return s[-1] / s[0]


def __print_time(old_time_stamp):
    """ For timing. """

    new_time_stamp = time.perf_counter()
    # Comment the line below to turn off time printing
    print(f"\t\t\t\t\t\t time: {np.round(new_time_stamp - old_time_stamp,5)}")
    return new_time_stamp


if __name__ == '__main__':
    """ Driver code for debugging and testing """
    train_x = np.random.random_sample((20, 2)) * 2 - 1
    #train_y = np.sum(np.sin(train_x), axis=1)
    train_y = np.sum(train_x**2+train_x, axis=1)
    # test_x = np.random.random_sample((20, 2)) * 2 - 1
    test_x = np.random.random_sample((4,2)) *2 -1
    #test_y = np.sum(np.sin(test_x), axis=1)
    test_y = np.sum(test_x**2+test_x, axis=1)
    ots = time.perf_counter() # ots = old time stamp
    print(f"Percent convex hull: \t{convexHullPercent(train_x, test_x)}") 
    ots = __print_time(ots)
    print(f"Maximin distance: \t{maximin(train_x)}")
    ots = __print_time(ots)
    print(f"Minimax distance: \t{minimax(train_x)}")
    ots = __print_time(ots)
    print(f"Sampling density: \t{samplingDensity(train_x)}")
    ots = __print_time(ots)
    print(f"Sampling distances: \t{samplingDistances(train_x)}")
    ots = __print_time(ots)
    print(f"Centerd l2 discrep: \t{discrepancy(train_x)}")
    ots = __print_time(ots)
    print(f"E-optimality crit: \t{eOptimality(train_x)}")
    ots = __print_time(ots)
    print(f"Inverse condition: \t{invCondition(train_x)}")
    ots = __print_time(ots)
    print(f"Condition of covar: \t{covarianceCondition(train_x)}")
    ots = __print_time(ots)
