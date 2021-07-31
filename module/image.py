import cv2
import numpy as np
from .ConcaveHull import ConcaveHull

def apply_slip(img, slip=1, mode='v', amplitude=4):
    slip = int(slip)
    assert slip > 0, 'slip is lower than 1.'
    assert mode in ['v', 'h', 'd', 's'], "mode not in {['v', 'h', 'd', 's']}"
    # Init
    mov = 2*slip    
    # Vertical  
    if mode == 'v':
        img_i = img[:-mov, slip:-slip]
        img_f = img[+mov:, slip:-slip]
    # horizontal
    if mode == 'h':
        img_i = img[slip:-slip, :-mov]
        img_f = img[slip:-slip, +mov:]
    # Diagonal
    if mode == 'd':
        img_i = img[+mov:, +mov:]
        img_f = img[:-mov, :-mov]
    # Skew Diagonal
    if mode == 's':
        img_i = img[+mov:, :-mov]
        img_f = img[:-mov, +mov:]

    img = img_f/510. - img_i/510.
    img = amplitude * img + 0.5
    img = np.pad(img, (slip,slip), 'constant', constant_values=0.5)
    img = np.clip(img, 0, 1)
    return img

def diff_img(img, slip=1, amplitude=2):
    v = apply_slip(img, slip=slip, mode='v', amplitude=amplitude)
    h = apply_slip(img, slip=slip, mode='h', amplitude=amplitude)
    img = 255 * (v+h) / 2
    return img.astype(np.uint8)

def grad_img(img, amplitude=4):
    x, y = np.gradient(img)
    img = amplitude * np.sqrt(x**2 + y**2)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def xy_grad_img(img):
    x, y = np.gradient(img)
    x    = cv2.normalize(x, x, 0, 255, cv2.NORM_MINMAX)
    y    = cv2.normalize(y, y, 0, 255, cv2.NORM_MINMAX)
    return x.astype(np.uint8), y.astype(np.uint8)

def log_img(img):
    img = img / 255.
    img = np.log1p(img) / np.log(2)
    img = np.clip(img, 0, 1)
    img = 255 * img
    return img.astype(np.uint8)

def denoising(img):
    return cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7, searchWindowSize=21)

def min_max_normalize(img):
    return cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

def canny(img):
    img = denoising(img)
    img = min_max_normalize(img)
    img = cv2.Canny(img, 50, 250, apertureSize=3, L2gradient=False)
    return img

def get_boundary_points(img, bbox, return_bbox=False):
    ch = ConcaveHull()
    pts = np.stack(np.where(img != 0), axis=1)
    
    flag = True
    if len(pts) < 4:
        flag = False
    if len(np.unique(pts[:, 0])) < 2:
        flag = False
    if len(np.unique(pts[:, 1])) < 2:
        flag = False
    if return_bbox:
        flag = False
        
    if flag:
        ch.loadpoints(pts)
        ch.calculatehull()
        boundary_points = np.vstack(ch.boundary.exterior.coords.xy).T
    else:
        x1, y1, x2, y2 = bbox
        boundary_points = np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]])
        
    boundary_points = list(boundary_points.flatten())
    return boundary_points

def imread(file):
    img = cv2.imread(file)[:, :, 0]    
    img_slip = diff_img(img, slip=2, amplitude=2)
    img_grad = grad_img(img)
    
    img_out = np.stack([img, img_slip, img_grad], axis=2)
    return img_out.astype(np.uint8)