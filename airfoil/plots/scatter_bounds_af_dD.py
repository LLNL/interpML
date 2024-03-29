from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import sys

if len(sys.argv) > 1:
    d = int(sys.argv[1])
else:
    d = 4

if len(sys.argv) > 2:
    rtype = sys.argv[2]
else:
    rtype = "valid"

# Set plot styles
sns.set()
sns.set_context("talk", font_scale=3)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
palette_list = sns.color_palette(palette="tab10", n_colors=10)
width = 4.5
height = width
fontsize = 12
mpl.rcParams.update({
    'text.usetex': True,
    'font.size': fontsize,
    'figure.figsize': (width, height), 
    'figure.facecolor': 'white', 
    'figure.subplot.bottom': 0.125, 
    'figure.edgecolor': 'white',
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize
})

# Set the output file
OUTFILE = f"../results/airfoil_latent{d}D_{rtype}_summary_2024-02.csv"
LOGFILE = f"../results/airfoil_latent{d}D_{rtype}_log_2024-02.csv"

# Keys for plot legend
NAMES = ["Delaunay", "GP", "TPS RBF"]

# Load data in pandas
df = pd.read_csv(LOGFILE)

# Start subplots
fig, axs = plt.subplots(3)
axs[0].set_title("Delaunay bounds vs. error", size=fontsize)
axs[1].set_title("GP bounds vs. error", size=fontsize)
axs[2].set_title("TPS bounds vs. error", size=fontsize)
for i in range(3):
    axs[i].set_ylabel("relative error")

# Set the yscale
yscale = 0.7074132530397318

for j, ENTRY in enumerate(NAMES):
    # Filter relevant methods by row
    rows = df[df["Method"] == ENTRY]
    rows = rows[rows["InCvxHull"] == 0]
    # Read relevant columns from file
    error_vals = rows["Error"].values.tolist()
    error_vals = [float(ei.strip()[1:-1]) / yscale for ei in error_vals]
    bound_vals = rows["Bound"].values.tolist()
    bound_vals = [float(bi.strip()[1:-1]) / yscale for bi in bound_vals]
    # Plot performance curves
    if True:
        axs[j].scatter(bound_vals, error_vals,
                       color="r", marker="x",
                       label="extrap pts")
    # Filter relevant methods by row
    rows = df[df["Method"] == ENTRY]
    rows = rows[rows["InCvxHull"] == 1]
    # Read relevant columns from file
    error_vals = rows["Error"].values.tolist()
    error_vals = [float(ei.strip()[1:-1]) / yscale for ei in error_vals]
    bound_vals = rows["Bound"].values.tolist()
    bound_vals = [float(bi.strip()[1:-1]) / yscale for bi in bound_vals]
    # Plot performance curves
    if True:
        axs[j].scatter(bound_vals, error_vals,
                       color="b", marker="o",
                       label="interp pts")

    # Set x/y limits and scales
    axs[j].set_yscale("symlog")
    axs[j].set_xscale("symlog")
    axs[j].set_ylim([0, 1])
    axs[j].set_xlim([0, 100])
    axs[j].set_yticks([0, 0.5, 1.0],
                      labels=["0", "0.5", "1"])
    axs[j].set_xticks([0, 0.5, 1.0, 10, 100],
                      labels=["0", "0.5", "1", "10", "100"])

    # Plot reference line
    axs[j].plot([k / 10 for k in range(11)],
                [k / 10 for k in range(11)],
                "k--",
                label="ref line")

# Add legends and show
axs[1].legend(loc="upper right")
plt.tight_layout()
plt.show()

#plt.savefig(f"../figs/airfoil_latent{d}D_{rtype}_scatter.eps")
if rtype == "test":
    # Reset subplots
    plt.clf()
    fig, axs = plt.subplots(3)
    axs[0].set_title("Delaunay bounds vs. error", size=fontsize)
    axs[1].set_title("GP bounds vs. error", size=fontsize)
    axs[2].set_title("TPS bounds vs. error", size=fontsize)
    for i in range(3):
        axs[i].set_ylabel("relative error")

    for j, ENTRY in enumerate(NAMES):
        # Filter relevant methods by row
        rows = df[df["Method"] == ENTRY]
        rows = rows[rows["InCvxHull"] == 0]
        # Read relevant columns from file
        error_vals = rows["Error"].values.tolist()
        error_vals = [float(ei.strip()[1:-1]) / yscale for ei in error_vals]
        bound_vals = rows["Bound"].values.tolist()
        bound_vals = [float(bi.strip()[1:-1]) / yscale + 0.2 for bi in bound_vals]
        # Plot performance curves
        if True:
            axs[j].scatter(bound_vals, error_vals,
                           color="r", marker="x",
                           label="extrap pts")
        # Filter relevant methods by row
        rows = df[df["Method"] == ENTRY]
        rows = rows[rows["InCvxHull"] == 1]
        # Read relevant columns from file
        error_vals = rows["Error"].values.tolist()
        error_vals = [float(ei.strip()[1:-1]) / yscale for ei in error_vals]
        bound_vals = rows["Bound"].values.tolist()
        bound_vals = [float(bi.strip()[1:-1]) / yscale + 0.2 for bi in bound_vals]
        # Plot performance curves
        if True:
            axs[j].scatter(bound_vals, error_vals,
                           color="b", marker="o",
                           label="interp pts")
    
        # Set x/y limits and scales
        axs[j].set_yscale("symlog")
        axs[j].set_xscale("symlog")
        axs[j].set_ylim([0, 1])
        axs[j].set_xlim([0.2, 100])
        axs[j].set_yticks([0, 0.5, 1.0],
                          labels=["0", "0.5", "1"])
        axs[j].set_xticks([0.2, 0.5, 1.0, 10, 100],
                          labels=["0.2", "0.5", "1", "10", "100"])
    
        # Plot reference line
        axs[j].plot([k / 10 for k in range(11)],
                    [k / 10 for k in range(11)],
                    "k--",
                    label="ref line")
    
    # Add legends and show
    axs[1].legend(loc="upper right")
    plt.tight_layout()
    #plt.savefig(f"../figs/airfoil_latent{d}D_{rtype}_scatter_shifted.eps")
    plt.show()
