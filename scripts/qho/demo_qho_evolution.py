import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from quantum import QuantumHarmonicOscillator

# ========================
#   Your Configuration
# ========================
config = {
    "model": {
        "angular_freq": 1,
        "mass": 1,
    },
    "evolution": {
        "x_grid": np.arange(-10, 10, 0.001),
        "t_grid": np.arange(0, 10, 0.1),
        "x_init": 0,  # initial position of Dirac delta
        "n_max": 50,  # number of eigen-state terms being summed
    },
    "save_fig": {
        "save": True,  # True if save the figure as html
        "filepath": "qho_evolution.html",
    }
}

qho = QuantumHarmonicOscillator(**config["model"])

# wavefunction is np.array
# 1st index: time
# 2nd index: space
wavefunction = qho.compute_amplitude_evolution_delta_init(**config["evolution"])

# Create dataframe for plotly express
data = {
        "t": [],
        "x": [],
        "psi_square": [],  # probability density
    }

psi_square = abs(wavefunction) ** 2

for t_id, t in enumerate(config["evolution"]["t_grid"]):
    for x_id, x in enumerate(config["evolution"]["x_grid"]):
        data["t"].append(t)
        data["x"].append(x)
        data["psi_square"].append(psi_square[t_id, x_id])

df = pd.DataFrame(
    data=data
)

# |psi|^2 Animation
fig = px.scatter(df, x="x", y="psi_square", animation_frame="t",
                 labels=dict(x="x", psi_square=r"$|\Psi(x,t)|^{2}$")
                 )

# facilitation plot
x_init = config["evolution"]["x_init"]
fig.add_trace(go.Scatter(x=[x_init, x_init], y=[0, psi_square[0, :].max()], mode="lines", name="Initial Position"))


if config["save_fig"]["save"]:
    fig.write_html(config["save_fig"]["filepath"])

fig.show()
