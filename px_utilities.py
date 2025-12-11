import plotly.graph_objects as go
import numpy as np
import stretchmap_utilities as su
import sham_utilities
from plotly.subplots import make_subplots

style_target = "dash"
density_color = "blue"
pressure_color = "green"
soundspeed_color = "magenta"
dotsize = 8

# import plotly.io as pio

# pio.templates.default = "plotly_dark"


def get_rho_values(data, mpart):
    H = data["hpart"]
    Rho = mpart * (1.2 / H) ** 3
    return Rho


def format_inputparams(input_params):
    string = ""
    for key, value in input_params.items():
        if key in ["y0", "n"]:
            valuestr = value
        elif key in ["mtot_target"]:
            valuestr = f"{value:.3f}"
        elif isinstance(value, float):
            if value > 10 or value < 0.1:
                valuestr = f"{value:.1e}"
            else:
                valuestr = f"{value:.1f}"
        else:
            valuestr = value
        string += f"{key}: {valuestr} <br>"
    return string


def px_3d_and_rho(model, ctx, img_path, rhotarget, input_params, eos):

    data = ctx.collect_data()
    t = model.get_time()
    mpart = model.get_particle_mass()
    Pos = data["xyz"]

    R = np.linalg.norm(Pos, axis=1)

    mtot_target = input_params["mtot_target"]
    Np = Pos.shape[0]
    integral_profile = su.integrate_target(rhotarget)
    rho0 = mtot_target / integral_profile

    Rho = get_rho_values(data, mpart)

    haspressure = False
    if "pressure" in data:
        haspressure = True
        pressure_data = data["pressure"]
    # haspressure = False

    nrows = 2
    ncols = 2
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[
            [{}, {"type": "scatter3d", "rowspan": 2}],
            [{"secondary_y": haspressure}, {}],
        ],
        horizontal_spacing=0.15,
        vertical_spacing=0.05,
        column_widths=[0.6, 0.4],
        subplot_titles=("Density profile", "Soundspeed and Pressure profiles", ""),
    )

    rhotab = rho0 * rhotarget[1]
    rtab = rhotarget[0]
    # ! Density profile
    fig.add_trace(
        go.Scatter(
            x=R,
            y=Rho,
            mode="markers",
            name="rhoSPH",
            marker=dict(color=density_color, size=dotsize),
            opacity=0.6,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rtab,
            y=rhotab,
            mode="lines",
            name="Initial target",
            line=dict(color=density_color, dash=style_target),
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(range=[0, None], row=1, col=1)
    # fig.update_yaxes(range=[0, None], row=2, col=1, secondary_y=False)
    # fig.update_yaxes(range=[0, None], row=2, col=1, secondary_y=True)

    # ! 3D scatter
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
                colorbar=dict(
                    title="Density",
                    tickfont=dict(size=24),
                    xanchor="left",
                    exponentformat="e",
                ),
                cmin=0,
                cmax=np.max(Rho),
                opacity=0.6,
                size=3,
            ),
            showlegend=False,
        ),
        col=2,
        row=1,
    )

    Pfunc, csfunc = su.get_p_and_cs_func(eos)
    cs_data = data["soundspeed"]
    # ! Soundspeed profile
    fig.add_trace(
        go.Scatter(
            x=R,
            y=cs_data,
            mode="markers",
            marker=dict(color=soundspeed_color, size=dotsize),
            name="soundspeed",
            opacity=0.6,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    mask = rhotab != 0
    fig.add_trace(
        go.Scatter(
            x=rtab[mask],
            y=csfunc(rhotab[mask]),
            mode="lines",
            line=dict(color=soundspeed_color, dash=style_target),
            showlegend=False,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    # ! Pressure profile
    if haspressure:
        fig.add_trace(
            go.Scatter(
                x=R,
                y=pressure_data,
                mode="markers",
                marker=dict(color=pressure_color, size=dotsize),
                name="pressure",
                opacity=0.6,
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=rtab,
                y=Pfunc(rhotab),
                mode="lines",
                line=dict(color=pressure_color, dash=style_target),
                showlegend=False,
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

    fig.update_xaxes(title_text=r"$r / R_\odot$", row=1, col=1)

    for pos in ([1, 1], [2, 1]):
        row, col = pos
        fig.update_xaxes(
            row=row,
            col=col,
            title_font=dict(size=20),
            tickfont=dict(size=24),
            tickprefix=r"$",
            ticksuffix=r"$",
            exponentformat="e",
        )
        fig.update_yaxes(
            row=row,
            col=col,
            title_font=dict(size=20),
            tickfont=dict(size=24),
            tickprefix=r"$",
            ticksuffix=r"$",
            exponentformat="e",
        )
    fig.update_xaxes(row=1, col=1, title_text=r"$r/R_\odot$")
    fig.update_yaxes(
        row=1,
        col=1,
        title_text=r"$\rho$",
        title_font=dict(color=density_color),
        tickfont=dict(color=density_color),
    )
    fig.update_xaxes(row=2, col=1, title_text=r"$r/R_\odot$")
    fig.update_yaxes(
        row=2,
        col=1,
        title_text=r"$c_s$",
        title_font=dict(color=soundspeed_color),
        tickfont=dict(color=soundspeed_color),
    )
    if haspressure:
        fig.update_yaxes(
            row=2,
            col=1,
            title_text=r"$P$",
            title_font=dict(color=pressure_color),
            tickfont=dict(color=pressure_color),
            secondary_y=True,
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=""),
            yaxis=dict(title=""),
            zaxis=dict(title=""),
        )
    )

    fig.update_layout(
        height=860,
        width=1920,
        title_text=f"{img_path} t={t:.2e}",
        annotations=[
            dict(
                text=f"{Np} particles, {Np*mpart:.1e} solar masses",
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.25,
        ),
    )

    # fig.add_annotation(
    #     text="Density profile",
    #     xref="paper",
    #     yref="paper",
    #     x=0,  # left column center (approx)
    #     y=0.98,
    #     showarrow=False,
    #     font=dict(size=16),
    # )
    # fig.add_annotation(
    #     text="Soundspeed and Pressure profiles",
    #     xref="paper",
    #     yref="paper",
    #     x=0,  # right/top area (approx) adjust as required
    #     y=0.5,
    #     showarrow=False,
    #     font=dict(size=16),
    # )

    return fig


def update_px_3d_and_rho(fig, model, ctx, img_path, input_params):
    data = ctx.collect_data()
    t = model.get_time()
    mpart = model.get_particle_mass()

    Pos = data["xyz"]
    R = np.linalg.norm(Pos, axis=1)
    Rho = get_rho_values(data, mpart)

    Np = Pos.shape[0]
    mtot_target = input_params["mtot_target"]

    fig.data[0].update(
        x=R,
        y=Rho,
    )

    # Trace 2 : Scatter3D
    fig.data[2].update(
        x=Pos[:, 0],
        y=Pos[:, 1],
        z=Pos[:, 2],
        marker_color=Rho,
        # marker=dict(
        #     cmin=0,
        #     cmax=np.max(Rho),
        # ),
    )

    cs_data = data["soundspeed"]
    fig.data[3].update(
        x=R,
        y=cs_data,
    )

    # fig.update_yaxes(range=[0, 1.1 * np.max(cs_data)], row=2, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, None], row=2, col=1, secondary_y=False)

    if "pressure" in data:
        pressure_data = data["pressure"]
        fig.data[5].update(
            x=R,
            y=pressure_data,
        )
        fig.update_yaxes(
            # range=[0, 1.1 * np.max(pressure_data)], row=2, col=1, secondary_y=True
            range=[0, None],
            row=2,
            col=1,
            secondary_y=True,
        )

    fig.update_yaxes(range=[0, 1.1 * np.max(Rho)], row=1, col=1)
    arr1inds = R.argsort()
    sorted_R = R[arr1inds]
    sorted_Rho = Rho[arr1inds]
    est_mass = su.integrate_target(np.array([sorted_R, sorted_Rho]))
    print(f"{est_mass}")

    fig.update_layout(
        title_text=f"{img_path} t={t:.2e}",
        annotations=[
            dict(
                text=f"{Np} particles, {Np*mpart:.1e} solar masses",
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
    )

    return fig


def movie(pattern_png, filemp4, fps):
    import ffmpeg

    ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
        filemp4,
        vcodec="libx264",
        crf=23,
        preset="medium",
        r=fps,
        pix_fmt="yuv420p",
        movflags="faststart",
    ).overwrite_output().run()


def compute_fps(inputparams):
    """
    so that the duration of the movie is 4 sec
    """
    nb_dumps = inputparams["nb_dumps"]
    # tf = inputparams["tf"]
    return int(nb_dumps / 4)
