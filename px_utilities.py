import plotly.graph_objects as go
import numpy as np
import stretchmap_utilities
import sham_utilities
from plotly.subplots import make_subplots


def get_rho_values(data, mpart):
    H = data["hpart"]
    Rho = mpart * (1.2 / H) ** 3
    return Rho


def format_inputparams(input_params):
    string = ""
    for key, value in input_params.items():
        if type(value) == float:
            string += f"{key}: {value:.1e} <br>"
        else:
            string += f"{key}: {value} <br>"
    return string


def px_3d_and_rho(data, mpart, t, img_path, rhotarget, input_params):

    Pos = data["xyz"]
    R = np.linalg.norm(Pos, axis=1)

    Np = Pos.shape[0]
    mtot = Np * mpart
    integral_profile = stretchmap_utilities.integrate_target(rhotarget)
    rho0 = mtot / integral_profile

    Rho = get_rho_values(data, mpart)

    haspressure = False
    if "pressure" in data:
        haspressure = True
        pressure = data["pressure"]

        # figrho = go.Figure(layout_yaxis_range=[0, 1.1 * np.max(rho0 * ytarget)])
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": haspressure}, {"type": "scatter3d"}]],
        horizontal_spacing=0.15,
        column_widths=[0.6, 0.4],
    )

    fig.add_trace(
        go.Scatter(
            x=R,
            y=Rho,
            mode="markers",
            name="rhoSPH",
            marker=dict(color="black", size=12),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rhotarget[0], y=rho0 * rhotarget[1], mode="lines", name="Initial target"
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(range=[0, None], exponentformat="E", row=1, col=1)

    fig.add_trace(
        go.Scatter3d(
            x=Pos[:, 0],
            y=Pos[:, 1],
            z=Pos[:, 2],
            mode="markers",
            name="",
            marker=dict(
                color=Rho,
                colorscale="Viridis",
                colorbar=dict(title="Density", tickfont=dict(size=24), xanchor="left"),
                cmin=0,
                cmax=np.max(Rho),
                opacity=0.6,
                size=2,
            ),
            showlegend=False,
        ),
        col=2,
        row=1,
    )

    if haspressure:
        fig.add_trace(
            go.Scatter(x=R, y=pressure, mode="markers", name="pressure"),
            row=1,
            col=1,
            secondary_y=True,
        )

    fig.update_layout(
        height=860,
        width=1920,
        title_text=f"{img_path} t={t:.2e}",
        annotations=[
            dict(
                text=f"{Np} particles, {mtot:.1e} solar masses",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.03,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=12, color="gray"),
            ),
            dict(
                text=format_inputparams(input_params),
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.55,
                y=0.5,
                bordercolor="black",
                borderwidth=1,
            ),
        ],
        font=dict(family="Courier New, monospace"),
        xaxis=dict(
            title_font=dict(size=16, family="Courier New"),
            tickfont=dict(size=24, family="Courier New"),
            title=r"$r / R_\odot$",
        ),
        yaxis=dict(
            title_font=dict(size=16, family="Courier New"),
            tickfont=dict(size=24, family="Courier New"),
            title=r"$\rho$",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.25,
        ),
    )

    return fig


def update_px_3d_and_rho(fig, data, mpart, t, img_path, input_params):
    # 1. Calculer les nouvelles données (comme dans votre fonction originale)
    Pos = data["xyz"]
    R = np.linalg.norm(Pos, axis=1)
    Rho = get_rho_values(data, mpart)

    Np = Pos.shape[0]
    mtot = Np * mpart

    # Note: On recalcule Rgrid et ytarget uniquement si l'intervalle [rmin, rmax] change
    # Si rmax peut changer, vous devez recalculer et mettre à jour la Trace 1 (Target) également.
    # Pour l'instant, on suppose la Trace 1 fixe si l'intervalle est constant.

    # 2. Mise à jour des traces existantes

    # Trace 0 : Scatter 2D (rhoSPH)
    fig.data[0].update(
        x=R,
        y=Rho,
    )

    if "pressure" in data:
        pressure = data["pressure"]
        print(pressure)
        fig.data[3].update(
            x=R,
            y=pressure,
        )
        fig.update_yaxes(
            range=[0, 1.1 * np.max(pressure)], row=1, col=1, secondary_y=True
        )

    fig.update_yaxes(range=[0, 1.1 * np.max(Rho)], row=1, col=1, secondary_y=False)
    arr1inds = R.argsort()
    sorted_R = R[arr1inds]
    sorted_Rho = Rho[arr1inds]
    est_mass = stretchmap_utilities.integrate_target(np.array([sorted_R, sorted_Rho]))
    print(f"{est_mass}")

    # Trace 2 : Scatter3D
    fig.data[2].update(
        x=Pos[:, 0],
        y=Pos[:, 1],
        z=Pos[:, 2],
        marker_color=Rho,
        marker=dict(
            color=Rho,
            colorscale="Viridis",
            colorbar=dict(title="Density", tickfont=dict(size=24), xanchor="left"),
            cmin=0,
            cmax=np.max(Rho),
            opacity=0.6,
            size=2,
        ),
        # Utilisez marker_color pour mettre à jour la couleur
    )

    # Mise à jour des échelles des axes (si nécessaire, ici seulement y-axis 2D)
    # fig.update_yaxes(range=[0, np.max(Rho) * 1.1], row=1, col=1)
    # Pour ne pas recréer la figure, il est souvent plus simple de laisser la plage
    # de l'axe y s'adapter à la nouvelle donnée, ou la fixer à une valeur maximale.

    # 3. Mise à jour du titre et de l'annotation (qui sont spécifiques au temps t)
    fig.update_layout(
        title_text=f"{img_path} t={t:.2e}",
        annotations=[
            dict(
                text=f"{Np} particles, {mtot:.1e} solar masses",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.03,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=12, color="gray"),
            ),
            dict(
                text=format_inputparams(input_params),
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.55,
                y=0.5,
                bordercolor="black",
                borderwidth=1,
            ),
        ],
        # Conservez tous les autres paramètres de layout
    )

    return fig


def movie(pattern_png, filemp4, fps):
    import ffmpeg

    ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
        filemp4,
        vcodec="libx264",
        crf=18,
        preset="medium",
        r=fps,
        pix_fmt="yuv420p",
        movflags="faststart",
    ).overwrite_output().run()


def compute_fps(inputparams):
    """
    so that the duration of the movie is proportional to the duration of the run
    """
    nb_dumps = inputparams["nb_dumps"]
    tf = inputparams["tf"]
    return int((nb_dumps / tf) * 2 / 3)
