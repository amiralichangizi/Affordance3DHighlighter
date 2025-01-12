import clip
from src.utils import device
import torch
from torchvision import transforms

def get_clip_model(clipmodel="ViT-L/14"):
    """
    Loads and configures a CLIP model for text-guided 3D highlighting.

    Args:
        clipmodel (str): Name of the CLIP model to use. Default is "ViT-L/14".
                        Other options include "ViT-L/14@336px", "RN50x4", "RN50x16", "RN50x64"

    Returns:
        tuple: (clip_model, preprocess, resolution)
            - clip_model: The loaded CLIP model
            - preprocess: CLIP's preprocessing transform
            - resolution: The appropriate resolution for the model
    """

    # Load the CLIP model and move to appropriate device
    clip_model, preprocess = clip.load(clipmodel, device=device, jit=False)

    # Determine the appropriate resolution for the model
    resolution = 224  # Default resolution
    if clipmodel == "ViT-L/14@336px":
        resolution = 336
    elif clipmodel == "RN50x4":
        resolution = 288
    elif clipmodel == "RN50x16":
        resolution = 384
    elif clipmodel == "RN50x64":
        resolution = 448

    # Freeze the model parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    # Set model to evaluation mode
    clip_model.eval()

    return clip_model, preprocess, resolution

def encode_text(clip_model, prompt, device):
    # Encode the text using CLIP
    with torch.no_grad():
        text_tokens = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features


def balanced_transform(clip_mean,clip_std,resolution=224):

    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(0.95, 1.0)),  # Slight cropping
        transforms.RandomPerspective(fill=1, p=0.7, distortion_scale=0.3),  # Moderate perspective
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle color variation
        transforms.Normalize(clip_mean, clip_std)
    ])
    return augment_transform

def lighting_transform(clip_mean,clip_std,resolution=224):

    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(0.9, 1.0)),
        transforms.RandomPerspective(fill=1, p=0.7, distortion_scale=0.4),
        transforms.ColorJitter(
            brightness=0.2,  # More brightness variation
            contrast=0.2,  # More contrast variation
            saturation=0.1,  # Slight saturation changes
        ),
        transforms.RandomGrayscale(p=0.05),  # Occasional grayscale
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Slight blur
        transforms.Normalize(clip_mean, clip_std)
    ])
    return augment_transform

def viewpoint_transform(clip_mean,clip_std,resolution=224):

    augment_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,  # Rotation up to 15 degrees
            translate=(0.1, 0.1),  # Translation up to 10%
            scale=(0.9, 1.1),  # Scale variation ±10%
        ),
        transforms.RandomPerspective(fill=1, p=0.9, distortion_scale=0.6),  # Stronger perspective
        transforms.RandomHorizontalFlip(p=0.3),  # Occasional flipping for symmetrical objects
        transforms.Normalize(clip_mean, clip_std)
    ])
    return augment_transform

def default_transform(clip_mean,clip_std,resolution=224):

    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        transforms.Normalize(clip_mean, clip_std)
    ])
    return augment_transform

def get_all_transforms(clip_mean,clip_std,resolution=224):
    transforms_dict = {
        'balanced': balanced_transform(clip_mean,clip_std,resolution),
        'viewpoint': viewpoint_transform(clip_mean,clip_std,resolution),
        'lighting': lighting_transform(clip_mean,clip_std,resolution),
        'default': default_transform(clip_mean,clip_std,resolution),
    }
    return transforms_dict

# Parameter grid for experimentation
param_grid = {
    'balanced': {
        'crop_scale': [(0.95, 1.0), (0.9, 1.0)],
        'perspective_scale': [0.3, 0.4],
        'color_strength': [0.1, 0.2]
    },
    'viewpoint': {
        'rotation_degrees': [10, 15, 20],
        'translate_range': [(0.1, 0.1), (0.2, 0.2)],
        'perspective_scale': [0.5, 0.6]
    },
    'lighting': {
        'brightness': [0.1, 0.2],
        'contrast': [0.1, 0.2],
        'blur_sigma': [(0.1, 0.5), (0.2, 0.7)]
    }
}
#TODO (OPTIONAL)
# Function to create transform with specific parameters
def get_transform_with_params(strategy, params, resolution):
    # CLIP's normalization values
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    if strategy == 'balanced':
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=params['crop_scale']),
            transforms.RandomPerspective(fill=1, p=0.7, distortion_scale=params['perspective_scale']),
            transforms.ColorJitter(brightness=params['color_strength'], contrast=params['color_strength']),
            transforms.Normalize(clip_mean, clip_std)
        ])
    # Add similar for other strategies...


def setup_clip_transforms(resolution=224,augumentation_type = "default"):
    """
    Creates the transformation pipelines needed for CLIP processing.

    Args:
        resolution (int): The target resolution for the images (depends on CLIP model)

    Returns:
        tuple: (clip_transform, augment_transform)
    """
    # CLIP's normalization values
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    # Basic CLIP transform - just resize and normalize
    clip_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.Normalize(clip_mean, clip_std)
    ])

    # Augmentation transform - adds random perturbations
    augment_transform = get_all_transforms(clip_mean,clip_std,resolution)[augumentation_type]

    return clip_transform, augment_transform