import torch
import csv
import os

def compute_mIoU(pred_labels, gt_labels):
    """
    pred_labels, gt_labels: shape [N] in {0,1}.
    Return a single float for IoU.

    For binary segmentation: IoU = intersection/union.
    We can do a direct approach:
    """
    intersection = ((pred_labels == 1) & (gt_labels == 1)).sum().float()
    union = ((pred_labels == 1) | (gt_labels == 1)).sum().float()
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou.item()

def evaluate_affordance(pred_scores, gt_labels, threshold=0.5):
    """
    pred_scores: shape [N], float, in range [0,1] or logit
    gt_labels: shape [N], 0/1
    threshold: 0.5 or user-chosen

    Convert pred_scores to {0,1} mask, compute IoU.
    """
    bin_preds = (pred_scores >= threshold).long()
    return compute_mIoU(bin_preds, gt_labels)

def evaluate_full_shape_objects(data_entries, net, clip_model, renderer, device='cuda', output_csv='results.csv'):
    """
    Loop over the dataset entries, run the pipeline for each affordance type, compute mIoU, and save to CSV.

    data_entries: list of dict from FullShapeDataset
    net: your NeuralHighlighter model
    clip_model: loaded CLIP model
    renderer: MultiViewPointCloudRenderer
    device: 'cuda' or 'cpu'
    output_csv: filename to store results
    """
    # We'll store rows: [shape_id, shape_class, affordance, mIoU]
    results = []

    for entry in data_entries:
        shape_id = entry['shape_id']
        shape_class = entry['shape_class']
        coords = entry['coords']            # [N,3]
        labels_dict = entry['labels_dict']  # { aff_key: [N], ...}
        affordances = entry['affordances']

        # For each affordance in this shape, we do:
        for aff_type in affordances:
            gt_label = labels_dict[aff_type]  # shape [N], 0/1 float

            # 1) Create the CLIP text prompt
            prompt = f"A 3D render of a gray {shape_class} with highlighted {aff_type} region"
            # or "with highlighted {aff_type}."

            # 2) Encode the text
            from src.Clip.clip_model import encode_text
            text_features = encode_text(clip_model, prompt, device=device)

            # 3) Run net to get highlight probabilities
            # net(coords) -> shape [N,2], if you are using 2-class Softmax
            pred_class = net(coords)
            # pred_class[:,0] = highlight probability, or use argmax if it's reversed
            # But if you are using 2-class softmax, the highlight prob might be in pred_class[:,0].
            # Let's assume highlight = index 0:
            highlight_scores = pred_class[:, 0]  # shape [N]

            # 4) Evaluate mIoU
            # Convert ground truth from float to int if needed
            gt_binary = (gt_label > 0.5).long()
            iou_val = evaluate_affordance(highlight_scores, gt_binary, threshold=0.5)

            results.append([shape_id, shape_class, aff_type, iou_val])

    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["shape_id", "shape_class", "affordance", "mIoU"])
        for row in results:
            writer.writerow(row)

    print(f"Results saved to {output_csv}. Number of rows: {len(results)}")
