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



def setup_clip_transforms(resolution=224):
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
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        transforms.Normalize(clip_mean, clip_std)
    ])

    return clip_transform, augment_transform