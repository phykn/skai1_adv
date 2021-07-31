import torch
import torchvision
import timm
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from timm.loss import LabelSmoothingCrossEntropy

class MASK_MODEL(nn.Module):
    def __init__(self, num_channel=3, num_class=12, min_size=100, max_size=100, image_mean=None, image_std=None, pretrained=False):
        super(MASK_MODEL, self).__init__()
        self.mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(min_size   = min_size, 
                                                                            max_size   = max_size, 
                                                                            pretrained = pretrained, 
                                                                            image_mean = image_mean, 
                                                                            image_std  = image_std)
        # Change Channel
        self.mask_rcnn.backbone.body.conv1 = nn.Conv2d(num_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Box Predictor
        cls_in_features = self.mask_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(cls_in_features, num_class)

        # Mask Predictor
        mask_in_features = self.mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        self.mask_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in_features, 256, num_class)

    def forward(self, images, targets=None):
        '''
        input image type  : [(C, W, H)]
        input image range : [0, 1]
        '''
        if targets is None:
            output = self.mask_rcnn(images)
        else:
            output = self.mask_rcnn(images, targets)
        return output
    
class CLF_MODEL(nn.Module):
    def __init__(self, name='efficientnetv2_rw_s', num_channel=3, num_class=12, image_mean=None, image_std=None, smoothing=0.1, pretrained=True):
        super(CLF_MODEL, self).__init__()
        self.image_mean = image_mean
        self.image_std  = image_std
        self.smoothing  = smoothing
        
        # EfficientNet
        self.effi            = timm.create_model(name, pretrained=pretrained)
        self.effi.conv_stem  = nn.Conv2d(num_channel, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        in_features          = self.effi.classifier.in_features
        self.effi.classifier = nn.Linear(in_features=in_features, out_features=num_class, bias=True)
        
        # Loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        
        # Softmax
        self.softmax = nn.Softmax(dim=0)

    def forward(self, images, targets=None):
        inputs = torch.cat([self.normalize(image, image_mean=self.image_mean, image_std=self.image_std).unsqueeze(0) for image in images])
        output = self.effi(inputs)
        
        if targets is None:
            return [{'prob': self.softmax(o), 'label': torch.argmax(o)} for o in output]        
        else:
            labels = torch.cat([target['label'].unsqueeze(0) for target in targets])
            loss   = self.criterion(output, labels)
            acc    = torch.mean(1.0 * (torch.argmax(output, axis=1) == labels))
            return {'loss': loss, 'acc': acc}
            
    def normalize(self, image, image_mean, image_std):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        image_mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
        image_std  = torch.as_tensor(image_std, dtype=dtype, device=device)
        return (image - image_mean[:, None, None]) / image_std[:, None, None]