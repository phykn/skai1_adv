import os
import random
import numpy as np
import pandas as pd
from datetime import datetime

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_time(message='', new_line=True):
    if new_line:
        print('')
    print(f'Time: {str(datetime.now())} | {message}')

def get_batch_file_list(files, batch_size=32):
    batch_file_list, tmp = [], []
    for file in files:
        if len(tmp) < batch_size:
            tmp.append(file)
        else:
            batch_file_list.append(tmp)
            tmp = [file]

    if tmp not in batch_file_list:
        batch_file_list.append(tmp)
        
    return batch_file_list

def get_ts(num=1):
    assert num >= 1
    ts = [{'bc': False, 'flip': False, 'scale': 1.0}]
    if num > 1:
        for _ in range(1, num):
            t = {'bc': np.random.choice([True, False]), 'flip': np.random.choice([True, False]), 'scale': random.uniform(0.5, 1.0)}
            ts.append(t)
    return ts

def convert_to_df(file_path, pred, conf=0.5):
    # Load Data
    img_name = os.path.basename(file_path)
    labels   = pred['labels']
    boxes    = pred['boxes']
    t_scores = pred['t_scores']
    clf_prob = pred['clf_prob']

    # Select over conf
    index    = np.where(t_scores > conf)
    labels   = labels[index]
    boxes    = boxes[index]
    t_scores = t_scores[index]

    # Pass
    if len(labels) == 0:
        output = pd.DataFrame({
            'img_name'  : [img_name],
            'defect'    : [0],
            'defect_no' : [0],
            'score'     : [np.clip(clf_prob[0], 0.5, 1)],
            'x_center'  : [0],
            'y_center'  : [0],
            'width'     : [0],
            'height'    : [0]
        })
        output[['defect', 'defect_no']] = output[['defect', 'defect_no']].astype(int)
        output[['score']] = output[['score']].astype(float).round(4)
        
    # Defect
    else:
        # Box convert xyxy -> xywh
        boxes_xywh       = boxes.copy()
        boxes_xywh[:, 0] = (boxes[:, 2] + boxes[:, 0])/2
        boxes_xywh[:, 1] = (boxes[:, 3] + boxes[:, 1])/2
        boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0])
        boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1])

        # y_center sort 
        index      = np.argsort(boxes_xywh[:, 1])
        labels     = labels[index]
        defect_no  = np.array(range(len(index))) + 1
        t_scores   = t_scores[index]
        boxes_xywh = boxes_xywh[index]
        x_centers  = boxes_xywh[:, 0]
        y_centers  = boxes_xywh[:, 1]
        widths     = boxes_xywh[:, 2]
        heights    = boxes_xywh[:, 3]

        # output
        output = pd.DataFrame({
            'img_name'  : [img_name]*len(index),
            'defect'    : labels,
            'defect_no' : defect_no,
            'score'     : t_scores,
            'x_center'  : x_centers,
            'y_center'  : y_centers,
            'width'     : widths,
            'height'    : heights
        })
        output[['defect', 'defect_no']] = output[['defect', 'defect_no']].astype(int)
        output[['score']] = output[['score']].astype(float).round(4)
        output[['x_center', 'y_center', 'width', 'height']] = output[['x_center', 'y_center', 'width', 'height']].astype(float).round(2)
        
    return output