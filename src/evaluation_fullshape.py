# src/evaluation_fullshape.py

import torch
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from .prompt_strategies import generate_affordance_prompt

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


def grid_search_validation(val_dataset, net, clip_model, device='cuda',
                           strategies=('basic','functional','descriptive','action','interactive'),
                           thresholds=(0.3, 0.5, 0.7)):
    """
    Try combinations of strategies & thresholds on the ENTIRE validation set.
    Return (best_strategy, best_threshold) that yields highest average mIoU.
    """
    all_results = []
    for strategy in strategies:
        for th in thresholds:
            iou_sum = 0.0
            count   = 0
            for idx in range(len(val_dataset)):
                entry = val_dataset[idx]
                single_res = evaluate_single_object(entry, net, clip_model, threshold=th, strategy=strategy, device=device)
                for r in single_res:
                    iou_sum += r['mIoU']
                    count += 1
            mean_iou = iou_sum / count if count>0 else 0.0
            all_results.append((strategy, th, mean_iou))

    # pick the best (strategy, threshold)
    all_results.sort(key=lambda x: x[2], reverse=True)
    best_strategy, best_threshold, best_mean_iou = all_results[0]
    return best_strategy, best_threshold, best_mean_iou


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
