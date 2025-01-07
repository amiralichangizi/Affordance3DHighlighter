from itertools import islice

import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
import matplotlib.pyplot as plt
import numpy as np


class PointCloudRenderer:
    def __init__(self, image_size=512, radius=0.003, points_per_pixel=10, device="cuda"):
        self.device = device
        # Initialize renderer settings
        self.raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius,
            points_per_pixel=points_per_pixel
        )

    def setup_renderer(self, R=None, T=None):
        if R is None or T is None:
            R, T = look_at_view_transform(20, 10, 0)
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, znear=0.01)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(1, 1, 1))
        )
        return renderer

    @staticmethod
    def create_point_cloud(points, colors=None):
        if colors is None:
            colors = torch.ones_like(points)  # Default white color
        return Pointclouds(points=[points], features=[colors])


class MultiViewPointCloudRenderer:
    def __init__(self, image_size=512, base_dist=20, base_elev=10, base_azim=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.image_size = image_size
        self.base_dist = base_dist
        self.base_elev = base_elev
        self.base_azim = base_azim

        # Define the settings for rasterization
        self.raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=0.003,
            points_per_pixel=10
        )

        # Define all views relative to base view
        self.views = {
            'Default': (base_dist, base_elev, base_azim),
            'Y_90deg': (base_dist, base_elev, base_azim + 90),
            'Y_180deg': (base_dist, base_elev, base_azim + 180),
            'Y_-90deg': (base_dist, base_elev, base_azim - 90),
            'X_90deg': (base_dist, base_elev + 90, base_azim),
            'X_-90deg': (base_dist, base_elev - 90, base_azim),
        }

    @staticmethod
    def create_point_cloud(points, colors=None):
        if colors is None:
            colors = torch.ones_like(points)  # Default white color
        return Pointclouds(points=[points], features=[colors])

    def get_center_point(self, point_cloud):
        """Calculate the center point of the point cloud"""
        points = point_cloud.points_packed()
        center = torch.mean(points, dim=0)
        return center.unsqueeze(0)  # Add batch dimension

    def create_renderer(self, dist, elev, azim, center_point,background_color=(0,0,0)):
        """Create a renderer for specific camera parameters"""
        # Use the center point as the 'at' parameter
        R, T = look_at_view_transform(
            dist=dist,
            elev=elev,
            azim=azim,
            at=center_point,  # Look at the center of the point cloud
        )
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, znear=0.01)

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=background_color)
        )
        return renderer

    def render_all_views(self, point_cloud, n_views,background_color = (0,0,0)):
        """Render point cloud from all defined views"""
        images = {}

        # Calculate center point once
        center_point = self.get_center_point(point_cloud)

        if n_views > 6:
            n_views = 6

        for view_name, (dist, elev, azim) in islice(self.views.items(), n_views):
            renderer = self.create_renderer(dist, elev, azim, center_point,background_color)
            image = renderer(point_cloud)
            images[view_name] = image[0, ..., :3].cpu()

        return images

    def plot_all_views(self, images):
        """Plot all rendered views in a grid"""
        fig = plt.figure(figsize=(20, 12))

        for idx, (view_name, image) in enumerate(images.items(), 1):
            ax = fig.add_subplot(2, 3, idx)
            ax.imshow(image)
            ax.set_title(
                f'{view_name}\nDist={self.base_dist}, Elev={self.views[view_name][1]:.1f}°, Azim={self.views[view_name][2]:.1f}°')
            ax.axis('off')

        plt.tight_layout()
        return fig


def load_point_cloud(file_path, device):
    """Load point cloud data from file"""
    data = np.load(file_path)
    verts = torch.tensor(data['verts']).to(device)
    rgb = torch.tensor(data['rgb']).to(device)
    return Pointclouds(points=[verts], features=[rgb])


def render_point_cloud_views(point_cloud_file, image_size=512, if_plot=False,n_views=6):
    """Complete pipeline to load, render and display point cloud from all views"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load point cloud
    point_cloud = load_point_cloud(point_cloud_file, device)

    # Create renderer
    renderer = MultiViewPointCloudRenderer(
        image_size=image_size,
        base_dist=20,  # Your default view distance
        base_elev=10,  # Your default elevation
        base_azim=0,  # Your default azimuth
        device=device
    )

    # Render all views
    rendered_images = renderer.render_all_views(point_cloud=point_cloud, n_views=n_views)

    # Plot results
    if if_plot:
        fig = renderer.plot_all_views(rendered_images)
        plt.show()

    return rendered_images
