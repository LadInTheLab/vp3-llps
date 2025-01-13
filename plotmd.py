import plotly.graph_objects as go
import numpy as np

def plottrajectory(trajectory, mode='markers', markersize=8, linewidth=10, colorscale='jet', color_array = None):
    # Get values from the trajectory
    c_alpha_indices = trajectory.c_alpha_indices
    trajectory_array = trajectory.array[c_alpha_indices]
    color_array = color_array[c_alpha_indices]

    x_coords = [point[0] for point in trajectory_array[:, 1]]
    y_coords = [point[1] for point in trajectory_array[:, 1]]
    z_coords = [point[2] for point in trajectory_array[:, 1]]
        # Create a 3D scatter plot
    fig = go.Figure()
    scatter = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode=mode,
        marker=dict(
            size=8,
            color=z_coords,  # Color points based on Z coordinate for verification purposes
            colorscale='Viridis',
            opacity=0.8
        ),
        line=dict(
            color=z_coords,
            width=linewidth)
    )
    fig.add_trace(scatter)
    frames = [go.Frame(
        data=[go.Scatter3d(
            x=trajectory_array[:, frame_idx, 0],
            y=trajectory_array[:, frame_idx, 1],
            z=trajectory_array[:, frame_idx, 2],
            mode=mode,
            marker=dict(
                size=markersize,
                color=color_array[:, frame_idx] if color_array is not None else z_coords,
                colorscale=colorscale),
            line=dict(
                color=color_array[:, frame_idx] if color_array is not None else z_coords,
                width=linewidth,
                colorscale=colorscale)
        )],
        name=f'Frame {frame_idx}'

    ) for frame_idx in range(1, trajectory_array.shape[1])]

    # Add frames to the subplot
    fig.frames = frames

    # Define layout settings
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[np.min(trajectory_array[:, :, 0]), np.max(trajectory_array[:, :, 0])], visible=True),
            yaxis=dict(range=[np.min(trajectory_array[:, :, 1]), np.max(trajectory_array[:, :, 1])], visible=True),
            zaxis=dict(range=[np.min(trajectory_array[:, :, 2]), np.max(trajectory_array[:, :, 2])], visible=True),
            aspectmode='cube',
        ),
        xaxis_title="X Position (nm)",
        yaxis_title="Y Position (nm)",
        template="plotly_dark",
        showlegend=True,
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 10, 'redraw': True}, 'fromcurrent': True}],
                 'label': 'Play',
                 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
                                   'transition': {'duration': 0}}],
                 'label': 'Pause',
                 'method': 'animate'},
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top',
        }],
        sliders=[{
            'active': 1,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Frame:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
        }]
    )
    # Show the figure
    fig.write_html("dimer.html")
    fig.show()