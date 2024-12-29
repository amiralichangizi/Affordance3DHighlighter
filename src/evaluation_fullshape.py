import torch
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from prompt_strategies import generate_affordance_prompt

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
    evaluate affordance predictions against ground truth
    """
    bin_preds = (pred_scores >= threshold).long()
    return compute_mIoU(bin_preds, gt_labels)

def visualize_predictions(coords, pred_scores, gt_labels, shape_id, affordance, output_dir):
    """
    Create visualization of predicted affordance regions vs ground truth.
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                         c=gt_labels.cpu(), cmap='coolwarm')
    ax1.set_title('Ground Truth')
    plt.colorbar(scatter)
    
    # Predictions
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                         c=pred_scores.cpu(), cmap='coolwarm')
    ax2.set_title('Predictions')
    plt.colorbar(scatter)
    
    # Difference
    ax3 = fig.add_subplot(133, projection='3d')
    diff = (pred_scores >= 0.5).float() - gt_labels
    scatter = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                         c=diff.cpu(), cmap='coolwarm')
    ax3.set_title('Difference')
    plt.colorbar(scatter)
    
    plt.suptitle(f'Shape {shape_id} - {affordance}')
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'visualizations', 
                            f'{shape_id}_{affordance}_comparison.png'))
    plt.close()


def evaluate_full_shape_objects(data_entries, net, clip_model, renderer, device='cuda', output_dir='results3'):
    """
    Evaluation pipeline for affordance prediction. 
    It tests different prompting strategies and creates detailed analysis.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # here we store results for each prompting strategy
    strategies = ['basic', 'functional', 'descriptive', 'action', 'interactive']
    all_results = []
   
    for entry in data_entries:
        shape_id = entry['shape_id']
        shape_class = entry['shape_class']
        coords = entry['coords']
        labels_dict = entry['labels_dict']
        affordances = entry['affordances']
        
        for aff_type in affordances:
            gt_label = labels_dict[aff_type]
            gt_binary = (gt_label > 0.5).long()
            
            # Test each prompting strategy
            for strategy in strategies:
                # Generate prompt
                prompt = generate_affordance_prompt(shape_class, aff_type, strategy)
                
                # Encode text
                from src.Clip.clip_model import encode_text
                text_features = encode_text(clip_model, prompt, device=device)
                
                # Run prediction
                pred_class = net(coords)
                highlight_scores = pred_class[:, 0]
                
                # Compute IoU
                iou_val = evaluate_affordance(highlight_scores, gt_binary)
                
                # Store result
                result = {
                    'shape_id': shape_id,
                    'shape_class': shape_class,
                    'affordance': aff_type,
                    'strategy': strategy,
                    'mIoU': iou_val,
                    'prompt': prompt
                }
                all_results.append(result)
                
                # Create visualization for basic strategy
                if strategy == 'basic':
                    visualize_predictions(coords, highlight_scores, gt_binary,
                                       shape_id, aff_type, output_dir)
    
    # Save detailed results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Create analysis
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
    plt.bar(strategy_perf.index, strategy_perf['mean'])
    plt.errorbar(strategy_perf.index, strategy_perf['mean'], 
                yerr=strategy_perf['std'], fmt='none', color='black')
    plt.title('Mean IoU per Prompting Strategy')
    plt.ylabel('Mean IoU')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_performance.png'))
    plt.close()
    
    # Plot affordance performance
    aff_perf = df.groupby('affordance')['mIoU'].agg(['mean', 'std'])
    plt.figure(figsize=(10, 6))
    plt.bar(aff_perf.index, aff_perf['mean'])
    plt.errorbar(aff_perf.index, aff_perf['mean'], 
                yerr=aff_perf['std'], fmt='none', color='black')
    plt.title('Mean IoU per Affordance Type')
    plt.ylabel('Mean IoU')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'affordance_performance.png'))
    plt.close()
