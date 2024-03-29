import pandas as pd
import sys


# -------- Begin main -------- #

p = 1
if len(sys.argv) > 1:
    d = int(sys.argv[1])
else:
    d = 4

# Set the input file
LOGFILE = f"../results/airfoil_latent{d}D_test_log.csv"
log_df = pd.read_csv(LOGFILE)

# Start the full data log
LOGFIELDS = ["Method", "InCvxHull", "Bound", "Error", "Simp", "Weights"]

i_m = 0; e_m = 0.15; b_m = 0.15; simp_m = None; weights_m = None
for i, rowi in log_df.iterrows():
    try:
        ei = float(rowi["Error"])
    except ValueError:
        ei = float(rowi["Error"].strip()[1:-1])
    try:
        bi = float(rowi["Bound"])
    except ValueError:
        bi = float(rowi["Bound"].strip()[1:-1])
    if rowi["Method"] == "Delaunay" and rowi["InCvxHull"] == True and ei > bi:
        if ei > e_m:
    #if rowi["Method"] == "Delaunay":
    #    if i == 3:
            i_m = i
            e_m = ei
            b_m = bi
            simp_m = rowi["Simp"]; weights_m = rowi["Weights"]
        # print(rowi["Simp"], rowi["Weights"])

print(simp_m)
print(weights_m)
print(i_m)
print(e_m, b_m)
