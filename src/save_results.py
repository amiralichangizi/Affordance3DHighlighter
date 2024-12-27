import os
import numpy as np
import torch
import torchvision
import open3d as o3d
import pytorch3d.structures
import pytorch3d.renderer as pr


# Save results
def save_point_cloud_results(points, colors, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    o3d.io.write_point_cloud(output_path, pcd)



def save_renders(dir, i, rendered_images, name=None):
    if name is not None:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, name))
    else:
        render_dir = os.path.join(dir, 'renders')
        os.makedirs(render_dir, exist_ok=True)  # Create renders directory
        torchvision.utils.save_image(rendered_images, os.path.join(render_dir, f'iter_{i}.jpg'))


def save_results(net, points, point_cloud, prompt, output_dir, renderer,device):
    """
    Saves the results of the highlighting process with proper image formatting.

    Args:
        net: Trained neural network
        points: Point cloud coordinates
        point_cloud: PyTorch3D Pointclouds object
        prompt: Text prompt used
        output_dir: Directory to save results
        renderer: Point cloud renderer
    """
    with torch.no_grad():
        # Get highlight predictions
        pred_class = net(points)

        # Create colors for visualization
        highlight_color = torch.tensor([204 / 255, 1.0, 0.0]).to(points.device)
        base_color = torch.tensor([180 / 255, 180 / 255, 180 / 255]).to(points.device)
        colors = pred_class[:, 0:1] * highlight_color + pred_class[:, 1:2] * base_color

        # Create and render point cloud
        point_cloud = renderer.create_point_cloud(points, colors)
        rendered_images = renderer.render_all_views(point_cloud)
        # Convert dictionary of images to tensor
        rendered_tensor = []
        for name, img in rendered_images.items():
            rendered_tensor.append(img.to(device))
        rendered_tensor = torch.stack(rendered_tensor)

        # Convert rendered images to CLIP format
        rendered_images = rendered_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Convert to uint8 range [0, 255]
        rendered_images = (rendered_images * 255).clamp(0, 255).to(torch.uint8)

        # Save point cloud data
        np.savez(
            os.path.join(output_dir, 'highlighted_points.npz'),
            points=points.cpu().numpy(),
            colors=colors.cpu().numpy(),
            probabilities=pred_class.cpu().numpy()
        )

        # Save prompt
        with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt)

        # Save rendered image using torchvision
        torchvision.utils.save_image(
            rendered_images.float() / 255.0,  # Convert back to [0,1] range
            os.path.join(output_dir, 'final_render.png'),
            normalize=False  # We've already normalized the values
        )