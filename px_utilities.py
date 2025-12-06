import plotly.graph_objects as go
import numpy as np
import stretchmap_utilities
import sham_utilities
from plotly.subplots import make_subplots


def get_rho_values(data, mpart):
    H = data["hpart"]
    Rho = mpart * (1.2 / H) ** 3
    return Rho


def px_Pos(data, mpart):
    Pos = data["xyz"]
    x = Pos[:, 0]
    y = Pos[:, 1]
    z = Pos[:, 2]
    rho_values = get_rho_values(data, mpart)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    color=rho_values,
                    colorscale="Viridis",
                    colorbar=dict(title="$\\rho$", x=0.8),
                    cmin=0,
                    cmax=np.max(rho_values),
                ),
            )
        ]
    )
    # fig.show()
    return fig


def px_rho(data, mpart, rhotarget):
    Pos = data["xyz"]
    R = np.linalg.norm(Pos, axis=1)
    rmin = 0
    rmax = np.max(R)
    Rgrid = np.linspace(rmin, rmax, 200)

    Np = Pos.shape[0]
    mtot = Np * mpart
    integral_profile = stretchmap_utilities.integrate_profile(rhotarget, rmin, rmax, 3)
    rho0 = mtot / integral_profile
    ytarget = rhotarget(Rgrid)

    Rho = get_rho_values(data, mpart)

    figrho = go.Figure(layout_yaxis_range=[0, 1.1 * np.max(rho0 * ytarget)])
    figrho.add_trace(go.Scatter(x=R, y=Rho, mode="markers", name="rhoSPH"))
    figrho.add_trace(go.Scatter(x=Rgrid, y=rho0 * ytarget, mode="lines", name="Target"))
    figrho.update_layout(xaxis_title=r"$r$", yaxis_title=r"$\rho$")
    # figrho.show()
    return figrho


def format_inputparams(input_params):
    string = ""
    for key, value in input_params.items():
        string += f"{key}: {value} <br>"
    return string


def px_3d_and_rho(data, rhotarget, mpart, t, input_params, img_path):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{}, {"type": "scatter3d"}]],
        horizontal_spacing=0.15,
        column_widths=[0.6, 0.4],
    )

    Pos = data["xyz"]
    R = np.linalg.norm(Pos, axis=1)
    rmin = 0
    rmax = np.max(R)
    Rgrid = np.linspace(rmin, rmax, 200)

    Np = Pos.shape[0]
    mtot = Np * mpart
    integral_profile = stretchmap_utilities.integrate_profile(rhotarget, rmin, rmax, 3)
    rho0 = mtot / integral_profile
    ytarget = rhotarget(Rgrid)

    Rho = get_rho_values(data, mpart)

    # figrho = go.Figure(layout_yaxis_range=[0, 1.1 * np.max(rho0 * ytarget)])
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
        go.Scatter(x=Rgrid, y=rho0 * ytarget, mode="lines", name="Target"), row=1, col=1
    )
    fig.update_yaxes(range=[0, None], row=1, col=1)

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
            ),
            showlegend=False,
        ),
        col=2,
        row=1,
    )
    fig.update_layout(
        height=860,
        width=1920,
        title_text=f"{img_path} t={t:.2e}",
        annotations=[
            dict(
                text=f"{data["xyz"].shape[0]} particles",  # Le texte du sous-titre
                showarrow=False,  # Cache la flèche d'annotation
                xref="paper",
                yref="paper",
                x=0.5,  # Centre le texte horizontalement
                y=1.03,  # Position verticale (ajustez cette valeur)
                xanchor="center",
                yanchor="bottom",
                font=dict(size=12, color="gray"),  # Style du sous-titre
            )
        ],
        font=dict(family="Courier New, monospace"),
        xaxis=dict(
            title_font=dict(size=16, family="Courier New"),
            tickfont=dict(size=24, family="Courier New"),
        ),
        yaxis=dict(
            title_font=dict(size=16, family="Courier New"),
            tickfont=dict(size=24, family="Courier New"),
        ),
        legend=dict(
            orientation="h",  # Légende horizontale
            yanchor="bottom",  # Ancrage vers le bas
            y=1.02,  # Position verticale (au-dessus du graphique)
            xanchor="center",  # Ancrage horizontal centré
            x=0.25,  # Position horizontale (dans le graphique de gauche)
        ),
        # paper_bgcolor="black",
        # plot_bgcolor="#111111",
        # margin=dict(t=0, b=0, l=0, r=0),
    )
    fig.add_annotation(
        text=format_inputparams(input_params),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.55,
        y=0.5,
        bordercolor="black",
        borderwidth=1,
    )

    return fig


def px_hist(data):
    r = np.linalg.norm(data["xyz"], axis=1)
    fighist = go.Figure(data=[go.Histogram(x=r, nbinsx=r.shape[0])])
    # fighist.show()
    return fighist


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
    nb_dumps = inputparams["nb_dumps"]
    tf = inputparams["tf"]
    return int((nb_dumps / tf) * 2 / 3)
