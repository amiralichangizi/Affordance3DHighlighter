# src/evaluation_fullshape.py

import torch
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from .prompt_strategies import generate_affordance_prompt

from src.render.cloud_point_renderer import MultiViewPointCloudRenderer
from src.neural_highlighter import NeuralHighlighter
from src.Clip.clip_model import encode_text
from src.render.cloud_point_renderer import MultiViewPointCloudRenderer
from src.save_results import save_renders, save_results
from src.neural_highlighter import NeuralHighlighter
from src.Clip.loss_function import clip_loss
from src.Clip.clip_model import get_clip_model, encode_text, setup_clip_transforms
import tqdm

def optimize_point_cloud(points, clip_model, renderer, encoded_text, log_dir: str, **kwargs):
    num_iterations = kwargs.get('num_iterations', 1000)
    learning_rate = kwargs.get('learning_rate', 1e-4)
    depth = kwargs.get('depth', 5)
    width = kwargs.get('network_width', 256)
    n_views = kwargs.get("n_views", 4)
    n_augs = kwargs.get('n_augs', 1)
    clipavg = kwargs.get('clipavg', 'view')
    device = kwargs.get('device', 'cuda')

    # Initialize network and optimizer
    net = NeuralHighlighter(
        depth=depth,  # Number of hidden layers
        width=width,  # Width of each layer
        out_dim=2,  # Binary classification (highlight/no-highlight)
        input_dim=3,  # 3D coordinates (x,y,z)
        positional_encoding=False  # As recommended in the paper
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Set up the transforms
    clip_transform, augment_transform = setup_clip_transforms()

    # Training loop
    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()

        # Predict highlight probabilities
        pred_class = net(points)

        # Create colors based on predictions
        highlight_color = torch.tensor([204 / 255, 1.0, 0.0]).to(device)
        base_color = torch.tensor([180 / 255, 180 / 255, 180 / 255]).to(device)

        colors = pred_class[:, 0:1] * highlight_color + pred_class[:, 1:2] * base_color

        # Create and render point cloud
        point_cloud = renderer.create_point_cloud(points, colors)
        rendered_images = renderer.render_all_views(point_cloud=point_cloud, n_views=n_views)
        # Convert dictionary of images to tensor
        rendered_tensor = []
        for name, img in rendered_images.items():
            rendered_tensor.append(img.to(device))
        rendered_tensor = torch.stack(rendered_tensor)

        #Convert rendered images to CLIP format
        rendered_images = rendered_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        #print(rendered_images.shape)

        # Calculate CLIP loss
        loss = clip_loss(
            rendered_images=rendered_images,
            encoded_text=encoded_text,
            clip_transform=clip_transform,
            augment_transform=augment_transform,
            clip_model=clip_model,
            n_augs=n_augs,
            clipavg=clipavg
        )
        #print("Loss computation graph:")
        #print_grad_fn(loss)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.4f}")
            save_renders(log_dir, i, rendered_images)

    return net


def train_and_evaluate_shape(
    shape_entry, 
    clip_model, 
    strategy, 
    threshold, 
    device="cuda",
    num_iterations=200
):
    """
    1) Create a fresh net
    2) Use shape's first affordance + the 'strategy' to form the main prompt
    3) Train for 'num_iterations'
    4) Compute IoU across ALL shape's affordances with 'threshold'
    5) Return average IoU for this shape
    """
    shape_coords = shape_entry["coords"]
    shape_class = shape_entry["shape_class"]
    affs = shape_entry["affordances"]
    label_dict = shape_entry["labels_dict"]

    # Make sure coords is on GPU
    if not isinstance(shape_coords, torch.Tensor):
        shape_coords = torch.tensor(shape_coords, device=device)

    # Create the net
    net = NeuralHighlighter(depth=5, width=256, out_dim=2, input_dim=3).to(device)

    # Build the main prompt from shape's first affordance + selected strategy
    main_aff = affs[0]  # first affordance
    from .prompt_strategies import generate_affordance_prompt
    prompt_str = generate_affordance_prompt(shape_class, main_aff, strategy)
    text_feats = encode_text(clip_model, prompt_str, device=device)

    # Minimal renderer
    renderer = MultiViewPointCloudRenderer(
        image_size=256, base_dist=20, base_elev=10, device=device
    )

    # Train (optimize) for 'num_iterations'
    net = optimize_point_cloud(
        points=shape_coords,
        clip_model=clip_model,
        renderer=renderer,
        encoded_text=text_feats,
        log_dir="./val_tmp",
        num_iterations=num_iterations,
        device=device,
        n_views=2
    )

    # Evaluate IoU across all affordances
    from .evaluation_fullshape import compute_mIoU
    with torch.no_grad():
        pred_class = net(shape_coords)  # shape [N,2]
        highlight_scores = pred_class[:,0]

    shape_sum = 0.0
    c = 0
    for aff in affs:
        gt_bin = (label_dict[aff]>0.5).long()
        bin_pred = (highlight_scores >= threshold).long()
        iou_val = compute_mIoU(bin_pred, gt_bin)
        shape_sum += iou_val
        c += 1

    shape_mean = shape_sum/c if c>0 else 0.0
    return shape_mean


def grid_search_validation(
    val_dataset, 
    clip_model,
    device='cuda',
    strategies=('basic','affordance_specific','descriptive','action','interactive'),
    thresholds=(0.3, 0.5, 0.7),
    num_val_objects=3,
    num_iterations=200
):
    """
    For each (strategy, threshold), pick up to 'num_val_objects' shapes from val_dataset,
    train & evaluate each shape => average IoU => pick best combo.
    """
    import random
    
    # We'll pick 'num_val_objects' shapes from the val dataset for speed
    val_indices = list(range(min(num_val_objects, len(val_dataset))))
    best_strategy = None
    best_threshold = None
    best_iou = -1.0
    all_results = []

    print("[grid_search_validation] Starting shape-by-shape training on validation...")

    for strategy in strategies:
        for th in thresholds:
            print(f"  Trying strategy='{strategy}' threshold={th}")
            sum_iou = 0.0
            count = 0
            for idx in val_indices:
                shape_entry = val_dataset[idx]
                try:
                    shape_mean = train_and_evaluate_shape(
                        shape_entry, 
                        clip_model,
                        strategy,
                        th,
                        device=device,
                        num_iterations=num_iterations
                    )
                    sum_iou += shape_mean
                    count += 1
                except Exception as e:
                    print(f"    [Warning] Skipped shape idx={idx} due to error: {e}")
                    continue
            
            avg_iou = sum_iou / count if count>0 else 0.0
            all_results.append((strategy, th, avg_iou))
            print(f"    => Mean IoU={avg_iou:.3f} over {count} shapes")

            # Track best
            if avg_iou>best_iou:
                best_iou = avg_iou
                best_strategy = strategy
                best_threshold = th

    print("\n[grid_search_validation] Validation combos sorted by best IoU:")
    sorted_res = sorted(all_results, key=lambda x: x[2], reverse=True)
    for sres in sorted_res:
        print(f"    Strategy={sres[0]}, Th={sres[1]}, IoU={sres[2]:.3f}")

    print(f"\n[grid_search_validation] Best strategy={best_strategy}, threshold={best_threshold}, IoU={best_iou:.3f}")
    return best_strategy, best_threshold, best_iou



def test_phase_evaluation(
    test_dataset,
    clip_model,
    best_strategy,
    best_threshold,
    device='cuda',
    num_test_shapes=3,
    num_iterations=200
):
    """
    Evaluate the pipeline on 'num_test_shapes' from test_dataset 
    using the best strategy & threshold from validation.
    Returns the final average test IoU.
    """
    import random

    test_indices = list(range(min(num_test_shapes, len(test_dataset))))
    sum_test_iou = 0.0
    c = 0

    print(f"[test_phase_evaluation] Using strategy={best_strategy}, threshold={best_threshold}")

    for idx in test_indices:
        shape_entry = test_dataset[idx]
        try:
            shape_mean = train_and_evaluate_shape(
                shape_entry,
                clip_model,
                best_strategy,
                best_threshold,
                device=device,
                num_iterations=num_iterations
            )
            sum_test_iou += shape_mean
            c+=1
            print(f"  -> shape idx={idx}, shape mean IoU={shape_mean:.3f}")
        except Exception as e:
            print(f"  [Warning] Skipped shape idx={idx} due to error: {e}")
            continue
    
    final_test_iou = sum_test_iou/c if c>0 else 0.0
    print(f"[test_phase_evaluation] Final average test IoU = {final_test_iou:.3f} (over {c} shapes)")
    return final_test_iou


def compute_mIoU(pred_labels, gt_labels):
    intersection = ((pred_labels == 1) & (gt_labels == 1)).sum().float()
    union = ((pred_labels == 1) | (gt_labels == 1)).sum().float()
    if union == 0:
        return 0.0
    return (intersection / union).item()


def evaluate_one_affordance(net, clip_model, coords, gt_binary, shape_class, aff_type, strategy, threshold, device='cuda'):
    """
    Evaluate one object-affordance with a given strategy + threshold.
    """
    from src.Clip.clip_model import encode_text

    prompt = generate_affordance_prompt(shape_class, aff_type, strategy)
    # We do not necessarily need the text_features if not used in forward pass,
    # but you can encode them to remain consistent with pipeline logic
    text_feats = encode_text(clip_model, prompt, device=device)

    net.eval()
    with torch.no_grad():
        pred_class = net(coords)  # shape [N,2]
        highlight_scores = pred_class[:,0]  # index 0 => highlight prob
    bin_preds = (highlight_scores >= threshold).long()
    iou_val = compute_mIoU(bin_preds, gt_binary)
    return iou_val, prompt


def evaluate_single_object(data_entry, net, clip_model, threshold=0.5, strategy="basic", device='cuda'):
    """
    Evaluate a single object with the given threshold & prompt strategy.
    Returns: list of {affordance, IoU, prompt, shape_id, shape_class}
    """
    coords = data_entry['coords']
    shape_class = data_entry['shape_class']
    shape_id = data_entry['shape_id']
    affs = data_entry['affordances']
    labels_dict = data_entry['labels_dict']

    results = []
    for aff in affs:
        gt_binary = (labels_dict[aff]>0.5).long()
        iou_val, prompt = evaluate_one_affordance(
            net, clip_model, coords, gt_binary, shape_class, aff, strategy, threshold, device
        )
        results.append({
            'shape_id': shape_id,
            'shape_class': shape_class,
            'affordance': aff,
            'strategy': strategy,
            'threshold': threshold,
            'mIoU': iou_val,
            'prompt': prompt
        })
    return results





def evaluate_dataset(dataset, net, clip_model, strategy, threshold, device='cuda'):
    """
    Evaluate an entire dataset with the chosen strategy/threshold, compute average mIoU
    """
    iou_sum = 0.0
    count = 0
    all_rows = []
    for idx in range(len(dataset)):
        entry = dataset[idx]
        single_res = evaluate_single_object(entry, net, clip_model, threshold=threshold, strategy=strategy, device=device)
        for r in single_res:
            iou_sum += r['mIoU']
            count += 1
            all_rows.append(r)
    mean_iou = iou_sum / count if count>0 else 0.0
    return mean_iou, all_rows


def visualize_predictions(coords, highlight_scores, gt_labels, shape_id, affordance, output_dir):
    """
    Simple 3D scatter plot visualizing predictions vs. ground truth.
    (Downsample to avoid heavy plotting if needed.)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    coords_np = coords.cpu().numpy()
    pred_np = highlight_scores.cpu().numpy()
    gt_np = gt_labels.cpu().numpy()

    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(coords_np[:,0], coords_np[:,1], coords_np[:,2], c=gt_np, cmap='coolwarm', s=1)
    ax1.set_title("Ground Truth")
    
    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(coords_np[:,0], coords_np[:,1], coords_np[:,2], c=pred_np, cmap='coolwarm', s=1)
    ax2.set_title("Prediction Prob")

    ax3 = fig.add_subplot(133, projection='3d')
    diff = (pred_np>=0.5).astype(float) - gt_np
    sc3 = ax3.scatter(coords_np[:,0], coords_np[:,1], coords_np[:,2], c=diff, cmap='coolwarm', s=1)
    ax3.set_title("Difference")

    plt.suptitle(f"{shape_id} - {affordance}")
    out_vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(out_vis_dir, exist_ok=True)
    outpath = os.path.join(out_vis_dir, f"{shape_id}_{affordance}_comparison.png")
    plt.savefig(outpath)
    plt.close()