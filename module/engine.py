import torch
import numpy as np
from mean_average_precision import MetricBuilder
from .label_map import get_label_map

@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):    
    model.eval()
    
    true_total, pred_total = [], []
    for images, targets in dataloader:                
        # Prediction
        preds = model([image.to(device) for image in images])
        for pred in preds:
            # Load Data
            scores = pred['scores'].detach().cpu().numpy()
            labels = pred['labels'].detach().cpu().numpy()
            boxes  = pred['boxes'].detach().cpu().numpy()

            pred_one_image = []
            if len(scores) == 0:
                pred_one_image += [[0, 0, 0, 0, 0, 1]]
            else:
                for box, label, score in zip(boxes, labels, scores):
                    out = list(box) + [label] + [score]
                    pred_one_image += [out]
                    
            pred_total += [np.array(pred_one_image)]            
        # Target
        for target in targets:
            labels = target['labels'].detach().cpu().numpy()
            boxes  = target['boxes'].detach().cpu().numpy()    

            true_one_image = []
            for box, label in zip(boxes, labels):
                out = list(box) + [label] + [0] + [0]
                true_one_image += [out]        

            true_total += [np.array(true_one_image)]
            
    return true_total, pred_total

def _predict_over_conf(pred_total, conf=0.25):
    pred_total_conf = []
    for pred_one_image in pred_total:
        boxes  = pred_one_image[:, :4]
        labels = pred_one_image[:, 4]  
        scores = pred_one_image[:, 5]

        select = np.where(scores > conf)
        scores = scores[select]
        labels = labels[select]
        boxes  = boxes[select]

        pred_one_image = []
        for box, label, score in zip(boxes, labels, scores):
            out = list(box) + [label] + [score]
            pred_one_image += [out]

        pred_total_conf += [np.array(pred_one_image)]        
    return pred_total_conf

def print_mAP(true, pred, conf=0.25):
    # Get Prediction
    pred = _predict_over_conf(pred, conf=conf)
    
    # Metric
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=12)
    for p, t in zip(pred, true):
        metric_fn.add(p, t)

    # mAP
    iou_thresholds    = np.round(np.arange(0.5, 1.0, 0.05), 2)
    recall_thresholds = np.round(np.arange(0., 1.01, 0.01), 2)

    metric = metric_fn.value(iou_thresholds=iou_thresholds, recall_thresholds=recall_thresholds, mpolicy='soft')    
    labels = list(metric[list(metric.keys())[0]].keys())
    names  = [get_label_map()[label] for label in labels]

    mAP_classes = []
    for label in labels:
        APs = [metric[iou_threshold][label]['ap'] for iou_threshold in iou_thresholds]
        mAP_class = np.sum(np.array(APs) * iou_thresholds) / np.sum(iou_thresholds)
        mAP_classes.append(mAP_class)
    mAP_total = np.mean(mAP_classes[1:])
    
    # Print        
    print(f'Conf:{conf}')
    print('------------------------------')
    for name, mAP in zip(names[1:], mAP_classes[1:]):
        print(f'{name:22s}| {mAP:.4f}')
    print(f'{"mAP@IoU=[.50:.05:.95]":22s}| {mAP_total:.4f}')
    return mAP_total