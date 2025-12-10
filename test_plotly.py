import plotly.graph_objects as go
import numpy as np

f = go.Figure(
    go.Scatter3d(
        x=np.random.rand(20000), y=np.random.rand(20000), z=np.random.rand(20000)
    )
)
f.write_image("test.png")
