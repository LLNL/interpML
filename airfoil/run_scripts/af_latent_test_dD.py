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
LOGFILE = f"../results/airfoil_latent{d}D_test_log.csv"
OUTFILE = f"../results/airfoil_latent{d}D_test_summary.csv"

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
outarray = np.zeros(1, dtype=dtypes)

# Start the full data log
__new_csv(LOGFILE, ["Method", "InCvxHull", "Bound", "Error", "Simp", "Weights"])

# Print helpful updates
print("\n\nLoading new train/test split...\n\n")

# Load the data
dtype = np.dtype([("x", "f8", d), ("y", "f8", (p,))])
train = np.zeros(5553, dtype=dtype)
test = np.zeros(686, dtype=dtype)
train["x"] = np.load(f"../airfoil_latent/Dim_{d}/Latent_space_train.npy")
train["y"] = np.load(f"../airfoil_latent/Dim_{d}/Train_labels.npy")
test["x"] = np.load(f"../airfoil_latent/Test_Data/Dim_{d}/Latent_space_test.npy")
test["y"] = np.load(f"../airfoil_latent/Test_Data/Dim_{d}/test_labels.npy")

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

# Generate training set size
perc_train = 1
n_train = n_train_full

# Reduce training set
train_reduced = interpml.deDupe(train[:n_train])

# Get test pts in convex hull
in_hull = interpml.checkInHull(train_reduced["x"],
                               test["x"],
                               rescale_x=True)
outarray[0]["perc in hull"] = len(in_hull) / test["x"].shape[0]

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

# Log full test set
for k, (bd, ae) in enumerate(zip(approx_bounds, abs_errs)):
    if k in in_hull:
        __save_to_csv(LOGFILE, "Delaunay", [1, bd, ae, simps[k,:], weights[k,:]])
    else:
        __save_to_csv(LOGFILE, "Delaunay", [0, bd, ae, simps[k,:], weights[k,:]])

# Log summary data
outarray[0]["del weak bound"] = np.mean(approx_bounds)
outarray[0]["del true bound"] = np.mean(true_bounds)
outarray[0]["del max error"] = np.max(abs_errs)
outarray[0]["del mae"] = np.mean(abs_errs)
if len(in_hull) > 0:
    outarray[0]["del interp weak bound"] = np.mean(approx_bounds[in_hull])
    outarray[0]["del interp true bound"] = np.mean(true_bounds[in_hull])
    outarray[0]["del interp max error"] = np.max(abs_errs[in_hull])
    outarray[0]["del interp mae"] = np.mean(abs_errs[in_hull])

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

# Log full test set
for k, (bd, ae) in enumerate(zip(approx_bounds, abs_errs)):
    if k in in_hull:
        __save_to_csv(LOGFILE, "GP", [1, bd, ae, "", ""])
    else:
        __save_to_csv(LOGFILE, "GP", [0, bd, ae, "", ""])
# Log summary data
outarray[0]["gp bound"] = np.mean(approx_bounds)
outarray[0]["gp max error"] = np.max(abs_errs)
outarray[0]["gp mae"] = np.mean(abs_errs)
if len(in_hull) > 0:
    outarray[0]["gp interp bound"] = np.mean(approx_bounds[in_hull])
    outarray[0]["gp interp max error"] = np.max(abs_errs[in_hull])
    outarray[0]["gp interp mae"] = np.mean(abs_errs[in_hull])

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

# Log full test set
for k, (bd, ae) in enumerate(zip(approx_bounds, abs_errs)):
    if k in in_hull:
        __save_to_csv(LOGFILE, "TPS RBF", [1, bd, ae, "", ""])
    else:
        __save_to_csv(LOGFILE, "TPS RBF", [0, bd, ae, "", ""])
# Log summary data
outarray[0]["rbf bound"] = np.mean(approx_bounds)
outarray[0]["rbf max error"] = np.max(abs_errs)
outarray[0]["rbf mae"] = np.mean(abs_errs)
if len(in_hull) > 0:
    outarray[0]["rbf interp bound"] = np.mean(approx_bounds[in_hull])
    outarray[0]["rbf interp max error"] = np.max(abs_errs[in_hull])
    outarray[0]["rbf interp mae"] = np.mean(abs_errs[in_hull])

# ReLU MLP error bounds
predictions = interpml.reluMlpInterp(train_reduced["x"],
                                     train_reduced["y"],
                                     test["x"],
                                     only_in_hull=False)
abs_errs = np.abs(predictions - test["y"])

# Log full training
for k, (bd, ae) in enumerate(zip(approx_bounds, abs_errs)):
    if k in in_hull:
        __save_to_csv(LOGFILE, "ReLU MLP", [1, "nan", ae, "", ""])
    else:
        __save_to_csv(LOGFILE, "ReLU MLP", [0, "nan", ae, "", ""])
# Log summary data
outarray[0]["relu mlp max error"] = np.max(abs_errs)
outarray[0]["relu mlp mae"] = np.mean(abs_errs)
if len(in_hull) > 0:
    outarray[0]["relu mlp interp max error"] = np.max(abs_errs[in_hull])
    outarray[0]["relu mlp interp mae"] = np.mean(abs_errs[in_hull])


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
        outdict[name].append(outarray[name][0])
        print(f"{outarray[name][0]:7.5f}")
    print()
print()

# Save the dictionary of means
df = pd.DataFrame(outdict)
df.to_csv(OUTFILE)
