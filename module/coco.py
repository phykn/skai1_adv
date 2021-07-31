import json
import numpy as np
import pandas as pd
from .label_map import get_label_map

def get_id(img_name):
    out = img_name.split('.')[0]
    out = out.split('_')[1]
    out = int(out)
    return out

def get_name(index):
    return get_label_map()[index]

def coco_dataset(df, dst, img_names=None):
    df = df.copy()
    
    # Select images
    if img_names is not None:
        group = df.groupby('img_name')
        df = pd.concat([group.get_group(img_name) for img_name in img_names])
        
    # Make columns
    df['id']        = df['img_name'].apply(lambda x: get_id(x))
    df['name']      = df['defect'].apply(lambda x: get_name(x))
    df['file_path'] = 'train/' + df['name'] + '/' + df['img_name']
    
    # Initialize Dataset
    coco = {}

    # COCO images
    coco['images'] = []
    df_images = df[['img_name', 'id', 'file_path']].drop_duplicates()
    for img_name, id, file_path in df_images.values:
        out              = {}
        out['id']        = id
        out['file_name'] = img_name
        out['file_path'] = file_path
        out['width']     = 100
        out['height']    = 100
        coco['images'].append(out)
        
    # COCO annotations
    coco['annotations'] = []
    df_annot = df[['defect', 'x_center', 'y_center', 'width', 'height', 'id', 'boundary_points']]
    annot_id = 0
    for category_id, x_center, y_center, width, height, image_id, boundary_points in df_annot.values:
        if category_id != 0:
            annot_id += 1
            
            # Get BBox Coordinate
            x1 = np.clip(x_center - width/2,  0, 100)
            x2 = np.clip(x_center + width/2,  0, 100)
            y1 = np.clip(y_center - height/2, 0, 100)
            y2 = np.clip(y_center + height/2, 0, 100)

            out                 = {}
            out['segmentation'] = [[float(item) for item in boundary_points.split(' ')]]
            out['area']         = width * height
            out['category_id']  = int(category_id)
            out['bbox']         = [x1, y1, x2-x1, y2-y1] #xywh
            out['image_id']     = int(image_id)
            out['iscrowd']      = 0
            out['id']           = annot_id
            coco['annotations'].append(out)

    # COCO categories
    coco['categories'] = []
    for key, value in get_label_map().items():
        out                  = {}
        out['id']            = key
        out['name']          = value
        out['supercategory'] = 'defect' if value != 'Pass' else 'background'
        coco['categories'].append(out)

    # Save File
    with open(dst, 'w') as f:
        json.dump(coco, f)