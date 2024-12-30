import torch
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from .prompt_strategies import generate_affordance_prompt

def compute_mIoU(pred_labels, gt_labels):
    """
    Compute Mean Intersection over Union for binary segmentation.
    """
    intersection = ((pred_labels == 1) & (gt_labels == 1)).sum().float()
    union = ((pred_labels == 1) | (gt_labels == 1)).sum().float()
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou.item()

def evaluate_affordance(pred_scores, gt_labels, threshold=0.5):
    """
    Evaluate affordance predictions against ground truth.
    """
    bin_preds = (pred_scores >= threshold).long()
    return compute_mIoU(bin_preds, gt_labels)

def visualize_predictions(coords, pred_scores, gt_labels, shape_id, affordance, output_dir):
    """
    Create visualization of predicted affordance regions vs ground truth.
    """
    sampled_idx = torch.randperm(coords.size(0))[:1000]  # Downsample for visualization
    coords = coords[sampled_idx]
    pred_scores = pred_scores[sampled_idx]
    gt_labels = gt_labels[sampled_idx]

    fig = plt.figure(figsize=(15, 5))
    
    # Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(coords[:, 0].cpu(), coords[:, 1].cpu(), coords[:, 2].cpu(), 
                           c=gt_labels.cpu(), cmap='coolwarm')
    ax1.set_title('Ground Truth')
    plt.colorbar(scatter, ax=ax1)
    
    # Predictions
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(coords[:, 0].cpu(), coords[:, 1].cpu(), coords[:, 2].cpu(), 
                           c=pred_scores.cpu(), cmap='coolwarm')
    ax2.set_title('Predictions')
    plt.colorbar(scatter, ax=ax2)
    
    # Difference
    ax3 = fig.add_subplot(133, projection='3d')
    diff = (pred_scores >= 0.5).float() - gt_labels
    scatter = ax3.scatter(coords[:, 0].cpu(), coords[:, 1].cpu(), coords[:, 2].cpu(), 
                           c=diff.cpu(), cmap='coolwarm')
    ax3.set_title('Difference')
    plt.colorbar(scatter, ax=ax3)
    
    plt.suptitle(f'Shape {shape_id} - {affordance}')
    plt.tight_layout()
    
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'visualizations', f'{shape_id}_{affordance}_comparison.png'))
    plt.close()

def run_inference_in_chunks(net, coords, chunk_size=20000):
    """
    Perform inference on coordinates in chunks to avoid memory issues.
    """
    N = coords.size(0)
    pred_scores = torch.zeros(N, device=coords.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pred_scores[start:end] = net(coords[start:end])[:, 0]
    return pred_scores

def evaluate_strategy(net, clip_model, coords, gt_binary, shape_class, affordance, strategy, device):
    """
    Evaluate a single prompting strategy.
    """
    prompt = generate_affordance_prompt(shape_class, affordance, strategy)
    from src.Clip.clip_model import encode_text
    text_features = encode_text(clip_model, prompt, device=device)
    highlight_scores = run_inference_in_chunks(net, coords, chunk_size=20000)
    iou = evaluate_affordance(highlight_scores, gt_binary)
    return iou, prompt, highlight_scores

def evaluate_full_shape_objects(data_entries, net, clip_model, renderer, device='cuda', output_dir='results3'):
    """
    Full evaluation pipeline for affordance prediction.
    """
    os.makedirs(output_dir, exist_ok=True)
    strategies = ['basic', 'functional', 'descriptive', 'action', 'interactive']
    results_file = os.path.join(output_dir, 'detailed_results.csv')

    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['shape_id', 'shape_class', 'affordance', 'strategy', 'mIoU', 'prompt'])
        writer.writeheader()

        for entry in data_entries:
            shape_id = entry['shape_id']
            shape_class = entry['shape_class']
            coords = entry['coords']
            labels_dict = entry['labels_dict']
            affordances = entry['affordances']

            for aff_type in affordances:
                gt_label = labels_dict[aff_type]
                gt_binary = (gt_label > 0.5).long()

                for strategy in strategies:
                    iou, prompt, highlight_scores = evaluate_strategy(net, clip_model, coords, gt_binary, shape_class, aff_type, strategy, device)
                    writer.writerow({'shape_id': shape_id, 'shape_class': shape_class, 'affordance': aff_type, 'strategy': strategy, 'mIoU': iou, 'prompt': prompt})

                    if strategy == 'basic':
                        visualize_predictions(coords, highlight_scores, gt_binary, shape_id, aff_type, output_dir)

    df = pd.read_csv(results_file)
    analyze_results(df, output_dir)

def analyze_results(df, output_dir):
    """
    Analyze and visualize evaluation results.
    """
    # Average performance per strategy and affordance
    strategy_aff_perf = df.groupby(['strategy', 'affordance'])['mIoU'].agg(
        ['mean', 'std', 'count']).round(4)
    strategy_aff_perf.to_csv(os.path.join(output_dir, 'strategy_affordance_performance.csv'))
    
    # Overall strategy performance
    strategy_perf = df.groupby('strategy')['mIoU'].agg(
        ['mean', 'std', 'count']).round(4)
    strategy_perf.to_csv(os.path.join(output_dir, 'strategy_performance.csv'))
    
    # Plot strategy performance
    plt.figure(figsize=(10, 6))
    plt.bar(strategy_perf.index, strategy_perf['mean'], yerr=strategy_perf['std'], capsize=4)
    plt.title('Mean IoU per Prompting Strategy')
    plt.ylabel('Mean IoU')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_performance.png'))
    plt.close()
    
    # Plot affordance performance
    aff_perf = df.groupby('affordance')['mIoU'].agg(['mean', 'std'])
    plt.figure(figsize=(10, 6))
    plt.bar(aff_perf.index, aff_perf['mean'], yerr=aff_perf['std'], capsize=4)
    plt.title('Mean IoU per Affordance Type')
    plt.ylabel('Mean IoU')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'affordance_performance.png'))
    plt.close()

def evaluate_single_object(data_entry, net, clip_model, device='cuda'):
    """
    Evaluate a single object with its affordances.
    Args:
        data_entry (dict): Single object data from the dataset.
        net (torch.nn.Module): Neural highlighting model.
        clip_model: CLIP model for text features.
        device (str): Device for computation ('cuda' or 'cpu').

    Returns:
        list: Results with IoU for each affordance.
    """
    from src.Clip.clip_model import encode_text

    shape_id = data_entry['shape_id']
    shape_class = data_entry['shape_class']
    coords = data_entry['coords']        # [N,3] on GPU
    affordances = data_entry['affordances']
    label_dict = data_entry['labels_dict']

    net.eval()
    results = []

    with torch.no_grad():
        for aff in affordances:
            gt_binary = (label_dict[aff] > 0.5).long()  # [N], 0 or 1

            prompt = generate_affordance_prompt(shape_class, aff, strategy='basic')
            text_features = encode_text(clip_model, prompt, device=device)

            # Run neural highlighter
            pred_class = net(coords)
            highlight_scores = pred_class[:, 0]  # Index 0 = highlight

            # Compute IoU
            iou_val = evaluate_affordance(highlight_scores, gt_binary)

            results.append({
                'shape_id': shape_id,
                'shape_class': shape_class,
                'affordance': aff,
                'IoU': iou_val,
                'prompt': prompt
            })

    return results


def visualize_single_object(data_entry, net, clip_model, device='cuda', out_dir='output'):
    """
    Render a 3D scatter plot for a single object, color-coded by GT and predicted highlights.
    Args:
        data_entry (dict): Single object data from the dataset.
        net (torch.nn.Module): Neural highlighting model.
        clip_model: CLIP model for text features.
        device (str): Device for computation ('cuda' or 'cpu').
        out_dir (str): Directory to save visualizations.

    Returns:
        None
    """
    from src.Clip.clip_model import encode_text

    shape_id = data_entry['shape_id']
    shape_class = data_entry['shape_class']
    coords = data_entry['coords']
    affordances = data_entry['affordances']
    labels_dict = data_entry['labels_dict']

    os.makedirs(out_dir, exist_ok=True)
    net.eval()

    # Visualize the first affordance
    aff = affordances[0]
    prompt = generate_affordance_prompt(shape_class, aff, strategy='basic')
    text_features = encode_text(clip_model, prompt, device=device)

    with torch.no_grad():
        pred_class = net(coords)
        highlight_scores = pred_class[:, 0]  # [N]

    # Prepare for plotting
    coords_np = coords.cpu().numpy()
    highlight_np = highlight_scores.cpu().numpy()
    gt_bin = (labels_dict[aff] > 0.5).long().cpu().numpy()

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], 
                c=gt_bin, cmap='coolwarm', s=2)
    ax1.set_title("Ground Truth")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2],
                c=highlight_np, cmap='coolwarm', s=2)
    ax2.set_title("Prediction Probability")

    plt.suptitle(f"Shape {shape_id} - {shape_class} - {aff}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{shape_id}_vis.png"))
    plt.close()
