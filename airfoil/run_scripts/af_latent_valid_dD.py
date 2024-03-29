import csv
import interpml
import numpy as np
import pandas as pd
import sys


# -------- Helper functions -------- #


def __save_to_csv(fname, method, row):
    """ Write a row of results to the csv file "fname" """

    with open(fname, "a") as fp:
        csvwriter = csv.writer(fp)
        to_add = [method]
        for col in row:
            to_add.append(col)
        csvwriter.writerow(to_add)


def __new_csv(fname, title):
    """ Start a header for the csv file "fname" """

    with open(fname, "w") as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(title)


# -------- Begin main -------- #

p = 1
if len(sys.argv) > 1:
    d = int(sys.argv[1])
else:
    d = 4

# Set the output file
OUTFILE = f"../results/airfoil_latent{d}D_valid_summary.csv"
LOGFILE = f"../results/airfoil_latent{d}D_valid_log.csv"

# Keep track of summary stats
OUTFIELDS = ["perc in hull",
             "del weak bound", "del true bound", "del max error", "del mae",
             "gp bound", "gp max error", "gp mae",
             "rbf bound", "rbf max error", "rbf mae",
             "relu mlp max error", "relu mlp mae",
             "del interp weak bound", "del interp true bound",
             "del interp max error", "del interp mae",
             "gp interp bound", "gp interp max error", "gp interp mae",
             "rbf interp bound", "rbf interp max error", "rbf interp mae",
             "relu mlp interp max error", "relu mlp interp mae"]

# Construct the dtype
dtypes = []
for name in OUTFIELDS:
    dtypes.append((name, float))
outarray = np.zeros((1, 10), dtype=dtypes)

# Start the full data log
__new_csv(LOGFILE, ["Method", "InCvxHull", "Bound", "Error", "Simp", "Weights"])

# Print helpful updates
print("\n\nLoading new train/test split...\n\n")

# Load the data
dtype = np.dtype([("x", "f8", d), ("y", "f8", (p,))])
train = np.zeros(5553, dtype=dtype)
test = np.zeros(616, dtype=dtype)
train["x"] = np.load(f"../data/Dim_{d}/Latent_space_train.npy")
train["y"] = np.load(f"../data/Train_labels.npy")
test["x"]  = np.load(f"../data/Dim_{d}/Latent_space_val.npy")
test["y"]  = np.load(f"../data/Val_labels.npy")

# Randomize order of training folds
np.random.seed(0)
orderi = np.arange(train.shape[0])
np.random.shuffle(orderi)
train = train[orderi]

# Get problem dimensions
n_train_full = train.size
n_test = test.size

# Check for correctness
assert(test["x"].shape[1] == d)

# Summary stats
print("Problem dimensions:\n")
print(f"number of training points:   {n_train_full}")
print(f"number of testing points:    {n_test}")
print(f"input dimension:             {d}")
print(f"output dimension:            {p}")

# Try 10 different data sizes
for j in range(10):
    # Generate training set size
    perc_train = (j+1) / 10
    n_train = int((j+1) * n_train_full // 10)

    # Reduce training set
    train_reduced = interpml.deDupe(train[:n_train])

    # Get % in convex hull
    in_hull = interpml.checkInHull(train_reduced["x"],
                                   test["x"],
                                   rescale_x=True)
    outarray[0, j]["perc in hull"] = len(in_hull) / test["x"].shape[0]

    # Delaunay error bounds
    true_bounds = interpml.delaunayError(train_reduced["x"],
                                         train_reduced["y"],
                                         test["x"],
                                         only_in_hull=False,
                                         true_bound=True)
    approx_bounds = interpml.delaunayError(train_reduced["x"],
                                           train_reduced["y"],
                                           test["x"],
                                           only_in_hull=False,
                                           true_bound=False)
    predictions, simps, weights = interpml.delaunayInterp(train_reduced["x"],
                                                          train_reduced["y"],
                                                          test["x"],
                                                          only_in_hull=False,
                                                          return_weights=True)
    abs_errs = np.abs(predictions - test["y"])

    # Only log on full training sets
    if perc_train >= 0.999:
        for k, (bd, ae) in enumerate(zip(approx_bounds, abs_errs)):
            if k in in_hull:
                __save_to_csv(LOGFILE, "Delaunay", [1, bd, ae, simps[k,:], weights[k,:]])
            else:
                __save_to_csv(LOGFILE, "Delaunay", [0, bd, ae, simps[k,:], weights[k,:]])


    # Log data
    outarray[0, j]["del weak bound"] = np.mean(approx_bounds)
    outarray[0, j]["del true bound"] = np.mean(true_bounds)
    outarray[0, j]["del max error"] = np.max(abs_errs)
    outarray[0, j]["del mae"] = np.mean(abs_errs)

    # Delaunay in hull bounds
    outarray[0, j]["del interp weak bound"] = np.mean(approx_bounds[in_hull])
    outarray[0, j]["del interp true bound"] = np.mean(true_bounds[in_hull])
    outarray[0, j]["del interp max error"] = np.max(abs_errs[in_hull])
    outarray[0, j]["del interp mae"] = np.mean(abs_errs[in_hull])

    # GP error bounds
    approx_bounds = interpml.gpError(train_reduced["x"],
                                     train_reduced["y"],
                                     test["x"],
                                     only_in_hull=False)
    predictions = interpml.gpInterp(train_reduced["x"],
                                    train_reduced["y"],
                                    test["x"],
                                    only_in_hull=False)
    abs_errs = np.abs(predictions - test["y"])

    # Only log on full training sets
    if perc_train >= 0.999:
        for k, (bd, ae) in enumerate(zip(approx_bounds, abs_errs)):
            if k in in_hull:
                __save_to_csv(LOGFILE, "GP", [1, bd, ae, "", ""])
            else:
                __save_to_csv(LOGFILE, "GP", [0, bd, ae, "", ""])

    # Log data
    outarray[0, j]["gp bound"] = np.mean(approx_bounds)
    outarray[0, j]["gp max error"] = np.max(abs_errs)
    outarray[0, j]["gp mae"] = np.mean(abs_errs)

    # GP in hull bounds
    outarray[0, j]["gp interp bound"] = np.mean(approx_bounds[in_hull])
    outarray[0, j]["gp interp max error"] = np.max(abs_errs[in_hull])
    outarray[0, j]["gp interp mae"] = np.mean(abs_errs[in_hull])

    # RBF error bounds
    approx_bounds = interpml.rbfError(train_reduced["x"],
                                      train_reduced["y"],
                                      test["x"],
                                      only_in_hull=False)
    predictions = interpml.rbfInterp(train_reduced["x"],
                                     train_reduced["y"],
                                     test["x"],
                                     only_in_hull=False)
    abs_errs = np.abs(predictions - test["y"])

    # Only log on full training sets
    if perc_train >= 0.999:
        for k, (bd, ae) in enumerate(zip(approx_bounds, abs_errs)):
            if k in in_hull:
                __save_to_csv(LOGFILE, "TPS RBF", [1, bd, ae, "", ""])
            else:
                __save_to_csv(LOGFILE, "TPS RBF", [0, bd, ae, "", ""])

    # Log data
    outarray[0, j]["rbf bound"] = np.mean(approx_bounds)
    outarray[0, j]["rbf max error"] = np.max(abs_errs)
    outarray[0, j]["rbf mae"] = np.mean(abs_errs)

    # RBF in hull bounds
    outarray[0, j]["rbf interp bound"] = np.mean(approx_bounds[in_hull])
    outarray[0, j]["rbf interp max error"] = np.max(abs_errs[in_hull])
    outarray[0, j]["rbf interp mae"] = np.mean(abs_errs[in_hull])

    # ReLU MLP error bounds
    predictions = interpml.reluMlpInterp(train_reduced["x"],
                                         train_reduced["y"],
                                         test["x"],
                                         only_in_hull=False)
    abs_errs = np.abs(predictions - test["y"])

    # Only log on full training sets
    if perc_train >= 0.999:
        for k, (bd, ae) in enumerate(zip(approx_bounds, abs_errs)):
            if k in in_hull:
                __save_to_csv(LOGFILE, "ReLU MLP", [1, "nan", ae, "", ""])
            else:
                __save_to_csv(LOGFILE, "ReLU MLP", [0, "nan", ae, "", ""])

    # Log data
    outarray[0, j]["relu mlp max error"] = np.max(abs_errs)
    outarray[0, j]["relu mlp mae"] = np.mean(abs_errs)

    # ReLU MLP in hull bounds
    outarray[0, j]["relu mlp interp max error"] = np.max(abs_errs[in_hull])
    outarray[0, j]["relu mlp interp mae"] = np.mean(abs_errs[in_hull])

    # Show progress
    print()
    print(f"Train size: {n_train}", f"Train percent: {perc_train:.4f}")


# Print summary stats
outdict = {}
print("\n\nSummary of performance:")
for name, _ in dtypes:
    if name not in outdict.keys():
        outdict[name] = []
    print(name, end=":\t")
    for i in range(10):
        outdict[name].append(np.mean(outarray[name][:, i]))
        print(f"{np.mean(outarray[name][:, i]):7.5f}")
    print()
print()

# Save the dictionary of means
df = pd.DataFrame(outdict)
df.to_csv(OUTFILE)
