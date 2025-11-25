import plotly.graph_objects as go
import numpy as np
import stretchmap_utilities

def px_Pos(data):
    Pos = data["xyz"]
    x=Pos[:,0]
    y=Pos[:,1]
    z=Pos[:,2]
    fig = go.Figure(data=[go.Scatter3d(x=x,y=y,z=z,mode='markers')])
    fig.show()
    return fig

def px_rho(data, mpart, rhotarget):
    Pos = data["xyz"]
    H = data["hpart"]
    Np = Pos.shape[0]
    mtot = Np * mpart
    # h = 1.2*(mpart/rho)**1/3
    Rho = mpart*(1.2/H)**3
    R = np.linalg.norm(Pos, axis=1)
    rmin = np.min(R)
    rmax = np.max(R)
    Rgrid = np.linspace(rmin, rmax, 200)

    integral_profile = stretchmap_utilities.integrate_profile(rhotarget, rmin, rmax, 3)
    rho0 = mtot/integral_profile

    ytarget = rhotarget(Rgrid)
    if type(ytarget)==int:
        ytarget = np.ones(Rgrid.shape)

    figrho = go.Figure()
    figrho.add_trace(
        go.Scatter(
            x=R,
            y=Rho,
            mode='markers',
            name='rhoSPH (?)'
        ))
    figrho.add_trace(
        go.Scatter(
            x=Rgrid,
            y=rho0*ytarget,
            mode='lines',
            name='Target'
        )
    )
    figrho.show()
    return figrho

def px_hist(data):
    r = np.linalg.norm(data["xyz"], axis=1)
    fighist = go.Figure(data=[go.Histogram(x=r, nbinsx=r.shape[0])])
    fighist.show()
    return fighist
