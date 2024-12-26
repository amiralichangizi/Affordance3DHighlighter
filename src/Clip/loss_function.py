import torch

def clip_loss(rendered_images, encoded_text, clip_transform, augment_transform, clip_model, n_augs=5, clipavg="view"):
    """
    Calculates the CLIP-based loss between rendered images and text description.

    The loss measures how well the highlighted regions match the text description
    by comparing their CLIP embeddings. Lower loss means better alignment.

    Args:
        rendered_images: Tensor of rendered views of the mesh
        encoded_text: CLIP embedding of the target text description
        clip_transform: Basic CLIP preprocessing transform
        augment_transform: Transform for data augmentation
        clip_model: The CLIP model for computing embeddings
        n_augs: Number of augmentations to apply (default: 5)
        clipavg: Method for averaging CLIP scores ("view" or "embedding")

    Returns:
        torch.Tensor: The computed loss value
    """
    # If no augmentations requested, just use basic transform
    if n_augs == 0:
        # Apply CLIP's preprocessing transform
        clip_image = clip_transform(rendered_images)

        # Get image embeddings from CLIP
        encoded_renders = clip_model.encode_image(clip_image)

        # Normalize embeddings to lie on unit sphere
        encoded_renders = encoded_renders / encoded_renders.norm(dim=1, keepdim=True)

        # Average across views or compare each view individually
        if clipavg == "view":
            # Handle both single and multiple text embeddings
            if encoded_text.shape[0] > 1:
                # Multiple text embeddings: average both image and text embeddings
                loss = torch.cosine_similarity(
                    torch.mean(encoded_renders, dim=0),
                    torch.mean(encoded_text, dim=0),
                    dim=0
                )
            else:
                # Single text embedding: just average image embeddings
                loss = torch.cosine_similarity(
                    torch.mean(encoded_renders, dim=0, keepdim=True),
                    encoded_text
                )
        else:
            # Compare each view individually and average the similarities
            loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

    # If augmentations requested, apply them and average results
    else:
        loss = 0.0
        # Run multiple augmentations and average their losses
        for _ in range(n_augs):
            # Apply random augmentation transforms
            augmented_image = augment_transform(rendered_images)

            # Get embeddings for augmented images
            encoded_renders = clip_model.encode_image(augmented_image)
            encoded_renders = encoded_renders / encoded_renders.norm(dim=1, keepdim=True)

            # Calculate loss based on averaging method
            if clipavg == "view":
                if encoded_text.shape[0] > 1:
                    loss -= torch.cosine_similarity(
                        torch.mean(encoded_renders, dim=0),
                        torch.mean(encoded_text, dim=0),
                        dim=0
                    )
                else:
                    loss -= torch.cosine_similarity(
                        torch.mean(encoded_renders, dim=0, keepdim=True),
                        encoded_text
                    )
            else:
                loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

    return loss