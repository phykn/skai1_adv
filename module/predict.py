import torch
import numpy as np
import albumentations as A
from ensemble_boxes import weighted_boxes_fusion as WBF
from .image import imread

def batch_prediction(mask_model, clf_model, batch, 
                     t={'bc': False, 'flip': False, 'scale': 1.0}, 
                     input_img_size=200, output_img_size=100, 
                     m_weight=0.5, c_weight=0.5,
                     device='cpu'):
    # Check t
    assert t['bc'] in [True, False]
    assert t['flip'] in [True, False]
    t['scale'] = np.clip(t['scale'], 0.5, 1.0)
    
    # Set eval
    mask_model.eval()
    clf_model.eval()
    
    # Transform
    transform = A.Compose([
        A.Resize(input_img_size, input_img_size, p=1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=t['bc']),
        A.HorizontalFlip(p=t['flip']),
        A.Affine (scale=t['scale'], interpolation=0, fit_output=False, p=1),
    ])
    inv_transform = A.Compose([
        A.Affine (scale=1/t['scale'], interpolation=0, fit_output=False, p=1.0),
        A.HorizontalFlip(p=t['flip']),
        A.Resize(output_img_size, output_img_size, p=1),
    ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # Prediction
    images = [transform(image=imread(file))['image'] for file in batch]
    inputs = [torch.as_tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1) / 255. for image in images]
    with torch.no_grad():
        mask_out = [{k: v.detach().cpu().numpy() for k, v in m.items()} for m in mask_model(inputs)]
        clf_out  = [{k: v.detach().cpu().numpy() for k, v in c.items()} for c in clf_model(inputs)]
        
    # Fix Mask
    for m in mask_out:
        m['masks']=[np.where(mask[0]>0.5, 1, 0).astype(np.uint8) for mask in m['masks']]

    # Inverse Transform
    out_images, out_preds = [], []
    for image, m, c in zip(images, mask_out, clf_out):
        index = np.array(range(len(m['labels'])))
        inv_transformed = inv_transform(image  = image,
                                        masks  = m['masks'],
                                        bboxes = m['boxes'],
                                        labels = index)
               
        # Transformed Image
        image = inv_transformed['image']
        out_images.append(image)

        # Transformed Annotation
        index = np.array(inv_transformed['labels'])
        pred  = {}        
        if (len(index)==0) or (c['label'].item()==0):
            pred['labels']   = np.array([])
            pred['boxes']    = np.array([])
            pred['masks']    = np.array([])
            pred['m_scores'] = np.array([])
            pred['t_scores'] = np.array([])
            pred['clf_prob'] = c['prob']
        else:
            labels   = np.array(m['labels'])[index]     
            m_scores = np.array(m['scores'])[index]
            c_scores = c['prob'][labels]
            t_scores = m_scores * m_weight + c_scores * c_weight        # score recalculation
    
            pred['labels']   = labels
            pred['boxes']    = np.array(inv_transformed['bboxes'])
            pred['masks']    = np.array(inv_transformed['masks'])[index]
            pred['m_scores'] = m_scores
            pred['t_scores'] = t_scores
            pred['clf_prob'] = c['prob']

        # Check Length
        lengths = [len(pred['labels']), len(pred['boxes']), len(pred['masks']), len(pred['m_scores']), len(pred['t_scores'])]
        if len(np.unique(lengths)) > 1:
            print(lengths)
            raise ValueError

        out_preds.append(pred)
    return out_images, out_preds

def batch_prediction_tta(mask_model, clf_model, batch, 
                         input_img_size=192, output_img_size=100, ts=[{'flip': False, 'scale': 1.0}], 
                         m_weight=0.5, c_weight=0.5,
                         iou_thr=0.5, skip_box_thr=0.0001, device='cpu'):
    tta_preds = []
    for t in ts:
        batch_images, batch_preds = batch_prediction(mask_model, clf_model, batch,
                                                     t               = t, 
                                                     input_img_size  = input_img_size, 
                                                     output_img_size = output_img_size, 
                                                     m_weight        = m_weight,
                                                     c_weight        = c_weight,
                                                     device          = device)
        tta_images = batch_images
        tta_preds.append(batch_preds)

    batch_tta_preds = []
    for b in range(len(batch)):
        boxes_list  = []
        scores_list = []
        labels_list = []
        clf_probs   = []
        mask_list   = []
        for i in range(len(ts)):
            boxes    = tta_preds[i][b]['boxes'] / output_img_size
            scores   = tta_preds[i][b]['t_scores']
            labels   = tta_preds[i][b]['labels']
            clf_prob = tta_preds[i][b]['clf_prob']            
            if len(scores) != 0:
                masks    = tta_preds[i][b]['masks'] * scores.reshape(-1, 1, 1)
                mask     = np.mean(masks, axis=0)
            else:
                mask     = np.zeros((output_img_size, output_img_size))

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
            mask_list.append(mask)
            clf_probs.append(clf_prob)

        boxes, scores, labels = WBF(boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes    = boxes * output_img_size
        mask     = np.mean(mask_list, axis=0)
        clf_prob = np.mean(clf_probs, axis=0)

        output = {}
        output['labels']   = labels
        output['boxes']    = boxes
        output['masks']    = mask
        output['t_scores'] = scores
        output['clf_prob'] = clf_prob
        batch_tta_preds.append(output)
        
    return tta_images, batch_tta_preds