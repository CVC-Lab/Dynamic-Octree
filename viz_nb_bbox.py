import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)


def create_trajectory_plot(time_series_data, timestamp, selected_object=None):
    fig = go.Figure()
    
    # Plot each object's position
    for obj_id, position in time_series_data[timestamp].items():
        trace = go.Scatter3d(
            x=[position[0]], y=[position[1]], z=[position[2]],
            mode='markers+text',
            name=f"Object {obj_id}",
            marker=dict(size=5),
            text=[f"{obj_id}"],  # Label each object with its ID
            textposition="top center"
        )
        fig.add_trace(trace)
    
    # Visualize the octree nodes as bounding boxes
    for node in octree.nodes:
        if node is not None:
            x_bounds, y_bounds, z_bounds = node.calculate_node_bounds()
            # Create a 3D bounding box for the node
            fig.add_trace(go.Mesh3d(
                x=[x_bounds[0], x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0], x_bounds[0], x_bounds[1], x_bounds[1]],
                y=[y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0], y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0]],
                z=[z_bounds[0], z_bounds[0], z_bounds[0], z_bounds[0], z_bounds[1], z_bounds[1], z_bounds[1], z_bounds[1]],
                color='lightblue',
                opacity=0.2,
                name=f"Node {octree.nodes.index(node)}"
            ))
    
    # Highlight the selected object and draw its interaction radius
    if selected_object is not None:
        pos = time_series_data[timestamp][selected_object]
        interaction_distance = 200
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            name=f"Selected Object {selected_object}",
            marker=dict(size=10, color='red')
        ))
        # Create a sphere representing the interaction radius
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = pos[0] + interaction_distance * np.cos(u) * np.sin(v)
        y = pos[1] + interaction_distance * np.sin(u) * np.sin(v)
        z = pos[2] + interaction_distance * np.cos(v)
        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale='Blues'))

    fig.update_layout(
        title=f"Trajectories at Timestamp {timestamp}",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position"
        ),
        showlegend=True
    )
    return fig

app.layout = html.Div([
    html.H1("3D Trajectory Visualization"),
    dcc.Graph(id='trajectory-plot'),
    html.Label("Select Timestamp:"),
    dcc.Slider(
        id='timestamp-slider',
        min=min(time_series_data.keys()),
        max=max(time_series_data.keys()),
        step=1,
        value=min(time_series_data.keys()),
        marks={i: str(i) for i in range(min(time_series_data.keys()), max(time_series_data.keys()) + 1)}
    ),
    html.Label("Select Object:"),
    dcc.Dropdown(
        id='object-dropdown',
        options=[{'label': f"Object {obj_id}", 'value': obj_id} for obj_id in time_series_data[min(time_series_data.keys())].keys()],
        value=list(time_series_data[min(time_series_data.keys())].keys())[0]
    ),
    html.H3("Neighborhood List:"),
    html.Div(id='nblist-display')
])

@app.callback(
    Output('trajectory-plot', 'figure'),
    [Input('timestamp-slider', 'value'), Input('object-dropdown', 'value')]
)
def update_trajectory_plot(timestamp, selected_object):
    return create_trajectory_plot(time_series_data, timestamp, selected_object)

@app.callback(
    Output('object-dropdown', 'options'),
    Input('timestamp-slider', 'value')
)
def update_object_dropdown(timestamp):
    return [{'label': f"Object {obj_id}", 'value': obj_id} for obj_id in time_series_data[timestamp].keys()]

@app.callback(
    Output('nblist-display', 'children'),
    [Input('timestamp-slider', 'value'), Input('object-dropdown', 'value')]
)
def update_nblist_display(timestamp, selected_object):
    if selected_object is not None and timestamp in nblist_data:
        neighbors = nblist_data[timestamp].get(selected_object, [])
        return html.Ul([html.Li(f"Object {neighbor[0]} - Distance: {neighbor[1]:.2f}") for neighbor in neighbors])
    return "Select an object to see its neighborhood list."

if __name__ == '__main__':
    app.run_server(debug=True)
