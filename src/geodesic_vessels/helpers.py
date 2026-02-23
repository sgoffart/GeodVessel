import numpy as np
import random
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import json

# config files handlers

def check_config_format():
    pass

def config_handler(config_file):
    pass


# plots

def plot_components_and_extremities_3d(
    labeled_mask,
    extremities_centerline,
    voxel_size=(3, 1, 1),
    ext_color = "red",
    filename="components_centerline_extremities_plot",
    save=False
):
    """
    Plot components (colored) and extremities from centerline (red).
    Args:
        labeled_mask: 3D labeled array
        extremities_centerline: list of coordinates (N, 3)
        voxel_size: tuple for scaling coordinates
        filename: output HTML file name
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    
    fig = go.Figure()
    num_components = labeled_mask.max()
    color_list = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    while len(color_list) < num_components:
        color_list *= 2

    # Plot each component's voxels
    for comp_id in range(1, num_components + 1):
        component_voxels = np.argwhere(labeled_mask == comp_id)
        if component_voxels.shape[0] == 0:
            continue
        color = color_list[(comp_id - 1) % len(color_list)]
        component_voxels = component_voxels * voxel_size
        fig.add_trace(go.Scatter3d(
            x=component_voxels[:, 0], y=component_voxels[:, 1], z=component_voxels[:, 2],
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.3),
            name=f'Component {comp_id}'
        ))

    # Plot extremities from centerline (red dots)
    if extremities_centerline:
        arr = np.array(extremities_centerline) * voxel_size
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
            mode='markers',
            marker=dict(size=8, color=ext_color, opacity=1, symbol='circle'),
            name='Centerline Extremities'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title=f'3D: {filename} ({ext_color})',
        showlegend=True
    )
    fig.show()
    
    if save: 
        fig.write_html(f"/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/output/{filename}.html")
    return fig

def plot_multiple_geodesics_3d(labeled_mask, paths, voxel_size=(3,1,1)):
    """
    Plot all components and multiple geodesic paths in 3D.
    
    Args:
        labeled_mask: 3D labeled array
        paths: list of dicts returned by compute_geodesic_from_single_centroid_high_intensity
        voxel_size: tuple for scaling coordinates
    """
    fig = go.Figure()
    num_components = labeled_mask.max()
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    # Plot all components
    for comp_id in range(1, num_components + 1):
        component_voxels = np.argwhere(labeled_mask == comp_id)
        if component_voxels.shape[0] == 0:
            continue
        color = colors[(comp_id-1) % len(colors)]
        pts = component_voxels * voxel_size
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.2),
            name=f'Component {comp_id}'
        ))

    # Overlay all geodesic paths
    for i, res in enumerate(paths):
        if res is None:
            print(f"skip path {i} because there is no path")
            prin(res)
            continue
        path_indices = np.array(res['path_indices']) * voxel_size
        start = np.array(res['from_extremity']) * voxel_size
        end = np.array(res['to_voxel']) * voxel_size

        # Path line
        fig.add_trace(go.Scatter3d(
            x=path_indices[:,0], y=path_indices[:,1], z=path_indices[:,2],
            mode='lines+markers',
            line=dict(color='black', width=5),
            marker=dict(size=4, color='black'),
            name=f'Path {i+1}'
        ))

        # Start and end points
        fig.add_trace(go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode='markers',
            marker=dict(size=6, color='green'),
            name=f'Start {i+1}'
        ))
        fig.add_trace(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
            mode='markers',
            marker=dict(size=6, color='red'),
            name=f'End {i+1}'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            ),
            title='Multiple Geodesic Paths in 3D',
            showlegend=True,
            width=900, height=900
        )
    fig.show()
    return fig
    
def plot_extremities_pairs_3d(closest_pairs, labeled_crop, voxel_size=(3, 1, 1),filename = "output_all_pairs_plot"):
    """
    Plot the closest pairs between components in 3D.
    closest_pairs: list of ((i, coord_i), (j, coord_j), min_dist)
    labeled_crop: 3D labeled array
    voxel_size: tuple for scaling coordinates
    """
    fig = go.Figure()
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    # Plot each component's voxels
    num_components = labeled_crop.max()
    for comp_id in range(1, num_components + 1):
        component_voxels = np.argwhere(labeled_crop == comp_id)
        component_voxels = component_voxels * voxel_size
        fig.add_trace(go.Scatter3d(
            x=component_voxels[:, 0], y=component_voxels[:, 1], z=component_voxels[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors[(comp_id-1) % len(colors)], opacity=0.3),
            name=f'Component {comp_id}'
        ))

    # Plot closest pairs as lines and points
    for (i, coord_i), (j, coord_j), min_dist in closest_pairs:
        pt1 = np.array(coord_i) * voxel_size
        pt2 = np.array(coord_j) * voxel_size
        fig.add_trace(go.Scatter3d(
            x=[pt1[0], pt2[0]], y=[pt1[1], pt2[1]], z=[pt1[2], pt2[2]],
            mode='markers+lines',
            marker=dict(size=5, color='blue', opacity=1),
            line=dict(color='blue', width=4),
            name=f'Closest: {i}->{j} ({min_dist:.1f})'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='3D Closest Pairs Between Components',
        showlegend=True
    )
    fig.show()
    fig.write_html(f"/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/output/{filename}.html")
    return fig

def plot_tangent_lines_3d(
    labeled_mask,
    extremities_by_comp,
    tangents_by_comp,
    voxel_size=(3, 1, 1),
    half_length=10.0,
    line_color='orange',
    line_width=10,
    show_components=True,
    show_extremities=True,
    ext_color='red',                # NEW
    ext_size=2,                     # NEW
    filename="tangent_lines_3d",
    save=False):
    """
    Plot tangents as straight line segments centered on extremities (no markers).

    half_length: half of the line length in physical units (after voxel_size scaling).
    """
    fig = go.Figure()

    # add extremity markers
    if show_extremities and extremities_by_comp:
        vx = np.array(voxel_size, dtype=float)
        all_exts = [np.array(ext, dtype=float) * vx for exts in extremities_by_comp.values() for ext in exts]
        if len(all_exts) > 0:
            A = np.vstack(all_exts)
            fig.add_trace(go.Scatter3d(
                x=A[:, 0], y=A[:, 1], z=A[:, 2],
                mode='markers',
                marker=dict(size=ext_size, color=ext_color, opacity=1.0, symbol='circle'),
                name='Extremities'
            ))

    if show_components:
        num_components = int(labeled_mask.max())
        color_list = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
        while len(color_list) < num_components:
            color_list *= 2
        for comp_id in range(1, num_components + 1):
            comp_vox = np.argwhere(labeled_mask == comp_id)
            if comp_vox.size == 0:
                continue
            pts = comp_vox * np.array(voxel_size, dtype=float)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=color_list[(comp_id - 1) % len(color_list)], opacity=0.2),
                name=f'Component {comp_id}'
            ))

    vx = np.array(voxel_size, dtype=float)
    for comp_id, exts in extremities_by_comp.items():
        tangs = tangents_by_comp.get(comp_id, [])
        for ext, t in zip(exts, tangs):
            if t is None:
                continue
            p = np.array(ext, dtype=float) * vx
            t_phys = t * vx
            n = np.linalg.norm(t_phys)
            if n == 0:
                continue
            d = (t_phys / n) * half_length
            p0 = p - d
            p1 = p + d

            fig.add_trace(go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                name='tangent'
            ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title='3D Extremity Tangent Lines',
        showlegend=False
    )
    fig.show()
    if save:
        fig.write_html(f"/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/output/{filename}.html")
    return fig