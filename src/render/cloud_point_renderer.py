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
            compositor=AlphaCompositor()
        )
        return renderer

    @staticmethod
    def create_point_cloud(points, colors=None):
        if colors is None:
            colors = torch.ones_like(points)  # Default white color
        return Pointclouds(points=[points], features=[colors])
