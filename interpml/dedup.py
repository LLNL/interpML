""" Utilities for deduplicating redundant training points.

Contains one public function:
* ``dedup(ndarray: train, tol=1e-4, verbose=False) -> ndarray``.

Use the above function to average training points that differ by less
than the argument ``tol``.

"""


import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def deDupe(train, tol=1e-4, verbose=False):
    """ Remove duplicates from train up to the tolerance.

    Args:
        train (numpy structured array): Training set to delete duplicates
            from containing 2 keys:
             - "x" (ndarray): are the input locations -- points will be
               clustered with respect to their "x" distances
             - "y" (ndarray): means will be applied to points in "y"
               space, but based on clustering in "x" space
        tol (float, optional): Minimum allowed 2-norm distance between points,
            defaults to 1e-4
        verbose (bool, optional): Print extra info when True, defaults to
            False

    Returns:
        numpy.ndarray: De-duplicated training set, by clustering nearest
        neighbors and replacing each cluster with its mean

    """

    # Get metadata info
    ndims_x = train["x"].shape[1]
    ndims_f = train["y"].shape[1]
    colnames_x = [f"x{i+1}" for i in range(ndims_x)]
    colnames_full = colnames_x.copy()
    for i in range(ndims_f):
        colnames_full.append(f"y{i+1}")
    # Create flattened copy of full dataset for later
    train_flattened = np.zeros((train.shape[0], ndims_x + ndims_f))
    train_flattened[:, :ndims_x] = train["x"][:, :]
    train_flattened[:, ndims_x:] = train["y"][:, :]
    # Convert to pandas dataframe
    full_dataset = pd.DataFrame(data=train_flattened, columns=colnames_full)
    inputdf = pd.DataFrame(data=__rescale_x(train["x"]), columns=colnames_x)
    # Set the tolerance uniformly
    array_tol = tol * np.ones(train["x"].shape[1])
    s_tol = pd.Series(array_tol, index=inputdf.columns)
    # Cluster by nearest neighbors
    gndf = __group_neighbors(inputdf, s_tol, p=2, verbose=verbose)
    if verbose:
        cols = []
        groups = []
        for i, coli in enumerate(gndf['set_id']):
            # print(f"{i}, {coli}")
            cols.append(i)
            groups.append(coli)
        cols = np.asarray(cols)
        groups = np.asarray(groups)
        inds = np.argsort(groups)
        cols = cols[inds]
        groups = groups[inds]
        with open("test.csv", "w") as fp:
            old_gi = 0
            for gi, ci in zip(groups, cols):
                if gi > old_gi:
                    fp.write("\n")
                    old_gi = gi
                else:
                    fp.write(",")
                fp.write(f"{ci}")
        print("==> gndf = ")
        print(gndf)
        print("==> gndf.dup(set id) = ")
        print(gndf.duplicated(subset='set_id'))
        print("==> sum(gndf.dup(set id)) = ")
        print(sum(gndf.duplicated(subset='set_id')))
        print("==> fd[gndf.dup(set id)] = ")
        print(full_dataset[gndf.duplicated(subset='set_id') == False])
    full_dataset['set_id'] = gndf['set_id']
    # Group neighbors by clusters, then create a new point with the mean
    # of that cluster, then replace "full_dataset" with the re-assigned data
    full_dataset = full_dataset.groupby(['set_id']
                                        ).mean().reset_index(drop=True)
    if verbose:
        print("==> For tolerance = ", array_tol,
              " there are ", len(full_dataset), " distinct inputs.")
    # Convert the full (filtered) dataset back into a numpy array
    full_dataset_np = np.zeros(len(full_dataset),
                               dtype=[('x', '<f8', (ndims_x,)),
                                      ('y', '<f8', (ndims_f,))])
    nx = train["x"].shape[1]
    ny = train["y"].shape[1]
    for i, rowi in full_dataset.iterrows():
        full_dataset_np[i]['x'] = np.asarray([rowi[f"x{j+1}"]
                                              for j in range(nx)])
        full_dataset_np[i]['y'] = np.asarray([rowi[f"y{j+1}"]
                                              for j in range(ny)])
    return full_dataset_np


def __group_neighbors(df, tol, p=np.inf, show=False, verbose=False):
    """ Cluster points by nearest neighbor.

    Code based on modification from the following source:

    https://www.tutorialguruji.com/python/finding-duplicates-with-tolerance-and-assign-to-a-set-in-pandas/

    Args:
        df (pandas.DataFrame): Data set to cluster
        tol (pandas.Series): Tolerance for clustering
        p (int, optional): p-norm to use for distance calculations, defaults
            to numpy.infty
        show (bool, optional): Print the graph when True, defaults to False
        verbose (bool, optional): Print extra info when True, defaults to
            False

    Returns:
        pandas.DataFrame: Clustered data set

    """

    r = np.linalg.norm(np.ones(len(tol)), p)
    if verbose:
        print("==> Making kd tree")
    kd = KDTree(df[tol.index] / tol)
    if verbose:
        print("==> Done making kd tree")
    g = nx.Graph([
        (i, j)
        for i, neighbors in enumerate(kd.query_ball_tree(kd, r=r, p=p))
        for j in neighbors
    ])
    if verbose:
        print("==> Done making neighbor graph")
    if show:
        nx.draw_networkx(g)
    ix, id_ = np.array([
        (j, i)
        for i, s in enumerate(nx.connected_components(g))
        for j in s
    ]).T
    id_[ix] = id_.copy()
    return df.assign(set_id=id_)


def __rescale_x(train_x):
    """ Rescale training and testing points to improve conditioning.

    Args:
        train_x (numpy.ndarray): A 2d float array, where each row indicates a
            training point

    Returns:
        rescale_train (numpy.ndarray):
        The two train datasets, rescaled dimension-wise to fill
        the cube [-1, 1]

    """

    __shift__ = np.zeros(train_x.shape[1])
    __scale__ = np.ones(train_x.shape[1])
    __shift__[:] = np.min(train_x, axis=0)
    __scale__[:] = np.max(np.array([xi - __shift__[:]
                                    for xi in train_x]), axis=0)
    rescaled_train = np.array([(xi - __shift__) / __scale__
                               for xi in train_x]) * 2 - 1
    return rescaled_train


# --- Driver code below (for testing) --- #

if __name__ == "__main__":
    """ Load airfoil data and combine rows that are the same up to the tol. """

    remove_dupes = True

    # Load the data
    dtype = np.dtype([("x", "f8", 4), ("y", "f8", (1,))])
    train = np.zeros(5553, dtype=dtype)
    train["x"] = np.load("../airfoil/data/Dim_4/Latent_space_train.npy")
    train["y"] = np.load("../airfoil/data/Train_labels.npy")

    reduced_train = deDupe(train, verbose=True)

    # Save k folds
    np.save("Latent_space_train_reduced.npy", reduced_train["x"])
    np.save("Train_labels_reduced.npy", reduced_train["y"][:, 0])
