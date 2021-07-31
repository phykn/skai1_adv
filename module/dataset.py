import os
import cv2
import torch
import numpy as np
from scipy.stats import mode
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from .image import imread
from .label_map import get_label_map

class MASK_Dataset(Dataset):
    def __init__(self, path, transforms=None, root_folder='../data/'):
        self.coco        = COCO(path)
        self.image_ids   = list(self.coco.imgToAnns.keys())
        self.labels      = [self.get_label(self.coco, image_id) for image_id in self.image_ids]
        self.transforms  = transforms
        self.root_folder = root_folder

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id  = self.image_ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        file_path = self.coco.loadImgs(image_id)[0]['file_path']
        file_path = os.path.join(self.root_folder, file_path)
        width     = self.coco.loadImgs(image_id)[0]['width']
        height    = self.coco.loadImgs(image_id)[0]['height']

        # Image
        image = imread(file_path)

        # Annotation
        annot_ids   = self.coco.getAnnIds(imgIds=image_id)
        annots      = [annot for annot in self.coco.loadAnns(annot_ids) if annot['image_id']==image_id]

        # Box (xywh -> xyxy)
        boxes       = np.array([annot['bbox'] for annot in annots])
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # Label
        labels      = [annot['category_id'] for annot in annots]
        
        # Mask
        masks       = [self.coco.annToMask(annot).T for annot in annots]
        
        # Area
        area        = [annot['area'] for annot in annots]
        
        # Iscrowd
        iscrowd     = [annot['iscrowd'] for annot in annots]
        
        if self.transforms:
            index = np.array(range(len(labels)))
            transformed = self.transforms(image  = image,
                                          masks  = masks,
                                          bboxes = boxes,
                                          labels = index)   
            image  = transformed['image']
            boxes  = transformed['bboxes']
            
            # Select labels & masks
            index  = np.array(transformed['labels']) 
            labels = np.array(labels)[index]
            masks  = np.array(transformed['masks'])[index]
            
        target = {'boxes'  : boxes,
                  'masks'  : masks,
                  'labels' : labels,
                  'area'   : area,
                  'iscrowd': iscrowd} 
        
        # To Tensor
        image               = torch.as_tensor(image, dtype=torch.float32)
        image               = image.permute(2, 0, 1) / 255.
        target['image_id']  = torch.as_tensor([image_id], dtype=torch.int64)
        target['boxes']     = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['masks']     = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['labels']    = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area']      = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd']   = torch.as_tensor(target['iscrowd'], dtype=torch.uint8)
        return image, target
    
    def get_label(self, coco, image_id):
        annot_ids = coco.getAnnIds(imgIds=image_id)
        annots    = [annot for annot in coco.loadAnns(annot_ids) if annot['image_id']==image_id]
        labels    = [annot['category_id'] for annot in annots]
        label     = mode(labels).mode[0]
        return label
    
class CLF_Dataset(Dataset):
    def __init__(self, img_names, defects, transforms=None, root_folder='../data/'):
        self.img_names   = img_names
        self.labels      = defects
        self.transforms  = transforms
        self.root_folder = root_folder

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name  = self.img_names[idx]
        label     = self.labels[idx]
        file_path = f'{self.root_folder}train/{get_label_map()[label]}/{img_name}'

        # Image
        image = imread(file_path) 
        if self.transforms:
            transformed = self.transforms(image  = image)
            image  = transformed['image']            
        target = {'labels' : label}
        
        # To Tensor
        image           = torch.as_tensor(image, dtype=torch.float32)
        image           = image.permute(2, 0, 1) / 255
        target          = {}
        target['label'] = torch.tensor(label, dtype=torch.int64)
        return image, target