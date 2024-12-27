import torch
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes


def obj_to_pointcloud(obj_path, num_points=10000, device="cuda"):
    """
    Converts an OBJ file to a point cloud by sampling points on the mesh surface.

    Args:
        obj_path (str): Path to the .obj file
        num_points (int): Number of points to sample from the mesh
        device (str): Device to store tensors on ('cuda' or 'cpu')

    Returns:
        tuple: (points_tensor, point_cloud_obj)
            - points_tensor is a tensor of shape (N, 3) containing the sampled points
            - point_cloud_obj is a PyTorch3D Pointclouds object
    """
    # Check if CUDA is available if device is set to "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    device = torch.device(device)

    # Load the obj file
    verts, faces_idx, _ = load_obj(obj_path)

    # Convert to tensors and move to specified device
    verts = verts.to(device)
    faces = faces_idx.verts_idx.to(device)

    # Create a Meshes object
    mesh = Meshes(
        verts=[verts],
        faces=[faces]
    )

    # Sample points uniformly from the mesh surface
    points = sample_points_from_meshes(
        mesh,
        num_samples=num_points,
        return_normals=False
    )

    # The sampled points tensor has shape (1, num_points, 3)
    # Remove the batch dimension
    points = points.squeeze(0)

    # Initialize features/colors (default to gray)
    gray_color = torch.ones_like(points) * 0.7

    # Create Pointclouds object
    point_cloud = Pointclouds(
        points=[points],
        features=[gray_color]
    )

    return points, point_cloud


def visualize_pointcloud(point_cloud, output_path=None, show=True):
    """
    Visualizes the point cloud using matplotlib.

    Args:
        point_cloud: PyTorch3D Pointclouds object or tensor of points
        output_path (str, optional): Path to save the visualization
        show (bool): Whether to display the plot
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Extract points from Pointclouds object if necessary
    if isinstance(point_cloud, Pointclouds):
        points = point_cloud.points_packed().cpu().numpy()
    else:
        points = point_cloud.cpu().numpy()

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c='gray',
        s=1
    )

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Make the plot more visually appealing
    ax.grid(True)
    ax.view_init(elev=30, azim=45)

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


# Example usage:
if __name__ == "__main__":
    # Convert OBJ to point cloud
    obj_path = "path/to/your/model.obj"
    points, point_cloud = obj_to_pointcloud(
        obj_path,
        num_points=10000,
        device="cuda"
    )

    # Visualize the result
    visualize_pointcloud(point_cloud, "pointcloud_viz.png")