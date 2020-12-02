import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from quantum import QuantumHarmonicOscillator

frequency = 1
mass = 1
qho = QuantumHarmonicOscillator(angular_freq=frequency, mass=mass)


# Set variable ranges
modes = list(range(0, 3))
colours = ["red", "blue", "green"]

x_values = np.arange(-5, 5, 0.001)


fig = make_subplots(rows=1,
                    cols=2,
                    subplot_titles=("Probability Amplitude",
                                    "Probability Density",
                                    ),
                    )
fig.layout.xaxis.title = "x"
fig.layout.xaxis2.title = "x"
fig.layout.yaxis.title = r"$\psi_{n}(x)$"
fig.layout.yaxis2.title = r"$|\psi_{n}(x)|^{2}$"

for colour, mode in zip(colours, modes):
    eigen_state_amp = qho.compute_energy_eigen_state_amplitude(mode, x_values)
    eigen_state_pdf = np.power(eigen_state_amp, 2)

    fig.add_trace(
        go.Scatter(x=x_values,
                   y=eigen_state_amp,
                   mode='lines',
                   legendgroup=f"{mode}",
                   name=f"n = {mode}",
                   marker=dict(color=colour),
                   ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=x_values,
                   y=eigen_state_pdf,
                   mode='lines',
                   legendgroup=f"{mode}",
                   name=f"n = {mode}",
                   showlegend=False,
                   marker=dict(color=colour),
                   ),
        row=1, col=2
    )

fig.show()

# Uncomment if you want to save the result as HTML
# fig.write_html("qho_eigenstate.html")
