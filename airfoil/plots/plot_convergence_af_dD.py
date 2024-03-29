from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
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

# Set the data file
OUTFILE = f"../results/airfoil_latent{d}D_{rtype}_summary_2024-02.csv"
LOGFILE = f"../results/airfoil_latent{d}D_{rtype}_log_2024-02.csv"

# Set the plot styles
sns.set()
sns.set_context("talk", font_scale=3)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
palette_list = sns.color_palette(palette="tab10", n_colors=10)
width = 6.0
height = width/2.6
fontsize = 12
mpl.rcParams.update({
    'text.usetex': True,
    'font.size': fontsize,
    'figure.figsize': (width, height), 
    'figure.facecolor': 'white', 
    'figure.subplot.left': 0.1, 
    'figure.subplot.right': 0.7, 
    'figure.subplot.top': 0.95, 
    'figure.subplot.bottom': 0.125, 
    'figure.edgecolor': 'white',
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize
})

# Fields for summary stats
OUTFIELDS1 = {"Delaunay": ["del mae", "del weak bound"],
              "GP": ["gp mae", "gp bound"],
              "TPS RBF": ["rbf mae", "rbf bound"],
              "ReLU MLP": ["relu mlp mae"]
             }
OUTFIELDS2 = {"Delaunay": ["del interp mae", "del interp weak bound"],
             "GP": ["gp interp mae", "gp interp bound"],
             "TPS RBF": ["rbf interp mae", "rbf interp bound"],
             "ReLU MLP": ["relu mlp interp mae"]
             }

SHOWMLP = True
OUTFIELDS = OUTFIELDS1

# Keys for plot legend
names = ["Delaunay", "GP", "TPS RBF", "ReLU MLP"]
#colors = ["g", "r", "b", "c"]
colors = [palette_list[2], palette_list[1], palette_list[4], palette_list[6]]
lines_mae = ["-o", "-s", "-^", "-X"]
lines_bd = ["--o", "--s", "--^", "--X"]

# Load data in pandas
df = pd.read_csv(OUTFILE)

# Start subplots
fig, axs = plt.subplots(1)
#axs.set_title("Training set size vs. average error bounds and MAE",
#                 fontsize=fontsize)
axs.set_ylabel("relative MAE")

# Set the yscale
yscale = 0.7074132530397318

if rtype == "test":
    for mname in ["Delaunay", "GP", "TPS RBF", "ReLU MLP"]:
        if mname != "ReLU MLP":
            print(f"{mname} & {df[OUTFIELDS1[mname][0]].loc[9] / yscale : .2f} & " +
                  f"{df[OUTFIELDS1[mname][1]].loc[9] / yscale : .2f} & " +
                  f"{df[OUTFIELDS2[mname][0]].loc[9] / yscale : .2f} & " +
                  f"{df[OUTFIELDS2[mname][1]].loc[9] / yscale : .2f} " +
                  "\\\\")
        else:
            print(f"{mname} & {df[OUTFIELDS1[mname][0]].loc[9] / yscale : .2f} & " +
                  f"NA & {df[OUTFIELDS2[mname][0]].loc[9] / yscale : .2f} & " +
                  "NA \\\\")
    sys.exit()

# Gather performance stats
for mi, mname in enumerate(names):
    # Calculate performance curves
    n_train = [555 * i for i in range(1, 11)]
    mae_vals = df[f"{OUTFIELDS[mname][0]}"].values.tolist()
    mae_vals = [ei / yscale for ei in mae_vals]
    # Plot performance curves
    if SHOWMLP or mname != "ReLU MLP":
        axs.plot(n_train, mae_vals,
                 f"{lines_mae[mi]}",
                 color=colors[mi],
                 label=f"{mname} MAE")

# Set bounds
axs.set_ylim([0.0, 0.11])
axs.set_yticks([0.0, 0.01, 0.05, 0.1],
               labels=["0", "1e-2", "5e-2", "1e-1"])
axs.set_xlim([0, 5750])
xticks = [0, 1000, 2000, 3000, 4000, 5000, 5553]
axs.set_xticks(xticks,
               labels=["0", "1K", "2K", "3K", "4K", "5K", " $\qquad$ 5553"])

# Add legends and show
axs.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
#plt.savefig(f"../figs/airfoil_latent{d}D_{rtype}_summary_zoom.eps")

# Reset plot style
fig, axs = plt.subplots(1)
axs.set_ylabel("relative MAE")
sns.set()
sns.set_context("talk", font_scale=3)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
palette_list = sns.color_palette(palette="tab10", n_colors=10)
width = 6.0
height = width/2.6
fontsize = 12
mpl.rcParams.update({
    'text.usetex': True,
    'font.size': fontsize,
    'figure.figsize': (width, height), 
    'figure.facecolor': 'white', 
    'figure.subplot.left': 0.1, 
    'figure.subplot.right': 0.7, 
    'figure.subplot.top': 0.95, 
    'figure.subplot.bottom': 0.125, 
    'figure.edgecolor': 'white',
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize
})

# Adjust the yscale
axs.set_yscale("symlog", linthresh=0.1)

# Re-plot performance stats
for mi, mname in enumerate(names):
    # Calculate performance curves
    n_train = [555 * i for i in range(1, 11)]
    mae_vals = df[f"{OUTFIELDS[mname][0]}"].values.tolist()
    mae_vals = [ei / yscale for ei in mae_vals]
    # Plot performance curves
    if SHOWMLP or mname != "ReLU MLP":
        axs.plot(n_train, mae_vals,
                 f"{lines_mae[mi]}",
                 color=colors[mi],
                 label=f"{mname} MAE")
    # Calculate bounds
    n_train = [555 * i for i in range(1, 11)]
    if len(OUTFIELDS[mname]) > 1:
        err_vals = df[f"{OUTFIELDS[mname][1]}"].values.tolist()
        err_vals = [ei / yscale for ei in err_vals]
    else:
        err_vals = []
    # Plot error bounds
    if mname != "ReLU MLP":
        axs.plot(n_train[:len(err_vals)], err_vals,
                 f"{lines_bd[mi]}",
                 color=colors[mi],
                 label=f"{mname} bound")

# Set bounds
axs.set_ylim([0.0, 20])
axs.set_yticks([0.0, 0.1, 1.0, 10.0],
               labels=["0", "1e-1", "1", "10"])
axs.set_xlim([0, 5750])
xticks = [0, 1000, 2000, 3000, 4000, 5000, 5553]
axs.set_xticks(xticks,
               labels=["0", "1K", "2K", "3K", "4K", "5K", " $\qquad$ 5553"])

# Add legends and show
axs.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
#plt.savefig(f"../figs/airfoil_latent{d}D_{rtype}_summary.eps")
