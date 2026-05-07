import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import argparse
import os

from dataset import ChangeDetectionDataset
from model import UNet

class ConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        self.tn += cm[0]
        self.fp += cm[1]
        self.fn += cm[2]
        self.tp += cm[3]

    def get_metrics(self):
        epsilon = 1e-7
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)
        
        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': [[self.tn, self.fp], [self.fn, self.tp]]
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Change Detection Model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--weights', type=str, default='best_model.pth', help='Path to model weights')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Split to evaluate on')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    data_dir = config['data']['val_dir'] if args.split == 'val' else config['data']['test_dir']
    dataset = ChangeDetectionDataset(data_dir, split=args.split)
    loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    model = UNet(n_channels=config['model']['in_channels'], n_classes=config['model']['out_channels']).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Warning: {args.weights} not found. Evaluating with random weights.")

    model.eval()
    
    cm_tracker = ConfusionMatrix()

    print(f"Starting evaluation on {args.split} split...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().cpu().numpy()
            
            cm_tracker.update(targets.numpy(), preds)
            
    metrics = cm_tracker.get_metrics()
    
    print("\n--- Evaluation Results ---")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    cm = metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

if __name__ == '__main__':
    main()
