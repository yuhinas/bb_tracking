from torch.utils.data import Dataset
from glob import glob
import torchvision.transforms as tsf
import torch
import random
import os
import cv2
import warnings

from utils import yolo_utils, torch_utils


to_tensor = tsf.ToTensor()
to_grayscale = tsf.Grayscale(1)

def yolo_collator(batch):
    imgs = [b[0].unsqueeze(0) for b in batch]
    imgs = torch.cat(imgs)
    labels = [b[1] for b in batch]
    boxes = [b[2] for b in batch]
    return imgs, labels, boxes


class YOLODataset(Dataset):
    """
    YOLO line format:
        img_path obj1 obj2 obj3...
        obj: norm_cx,norm_cy,norm_w,norm_h
    Return 
        image, boxes
    """
    # augmentation config
    brightness = 0.3
    contrast = 0.3
    flip_mode = 'all'  # all, h, v
    
    def __init__(self, yolo_lines, config, is_train=False):
        self.data = yolo_lines
        self.is_train = is_train
        
        self.resize = tsf.Resize(config['input_size'][::-1])
        
        self._color_jitter = tsf.ColorJitter(
            brightness = self.brightness,
            contrast = self.contrast
        )
    
    def _random_flip(self, img, boxes):
        if self.flip_mode in ['all', 'h']:
            if random.randint(0, 1):
                img = tsf.functional.hflip(img)
                if len(boxes):
                    boxes[:, 0] = 1 - boxes[:, 0]
        if self.flip_mode in ['all', 'v']:
            if random.randint(0, 1):
                img = tsf.functional.vflip(img)
                if len(boxes):
                    boxes[:, 1] = 1 - boxes[:, 1]
        return img, boxes
    
    def __getitem__(self, index):
        path, labels, boxes = yolo_utils.decode_line(self.data[index])
        labels = torch.Tensor(labels).long()
        
        random.shuffle(boxes)
        boxes = torch.Tensor(boxes)
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = to_tensor(img) 
        
        if self.is_train:
            img, boxes = self._random_flip(img, boxes)
            img = self._color_jitter(img)
        
        img = to_grayscale(img)
        img = self.resize(img)
        
        return img, labels, boxes
    
    def __len__(self):
        return len(self.data)


class MouseDataset(Dataset):
    """
    Required directory configuration:
        data_dir
        ├─ img/
        └─ mask/
    
    Requiring the same name of image and mask.
    """
    # augmentation config
    angle = 90
    translate = 0.2
    scale = [0.7, 1.3]
    shear = 20
    brightness = 0.3
    contrast = 0.3

    def __init__(self, img_list, config, is_train=True):
        self.img_list = img_list
        self.mask_list = [p.replace('/img/', '/mask/') for p in img_list]
        self.is_train = is_train
        self.input_size = config['input_size']

        self.resize = tsf.Resize(config['input_size'][::-1])
        
        self._color_jitter = tsf.ColorJitter(
            brightness = self.brightness,
            contrast = self.contrast
        )
        
    def _random_affine(self, img, mask):
        a = random.uniform(-1, 1)*self.angle
        w = int(random.uniform(0, self.translate)*self.input_size[0])
        h = int(random.uniform(0, self.translate)*self.input_size[1])
        b = random.uniform(*self.scale)
        c = random.uniform(-1, 1)*self.shear
         
        img = tsf.functional.affine(
            img,
            angle = a,
            translate = [w, h],
            scale = b,
            shear = c
        )
        mask = tsf.functional.affine(
            mask,
            angle = a,
            translate = [w, h],
            scale = b,
            shear = c
        )
        
        if random.random() > 0.5:
            img = tsf.functional.hflip(img)
            mask = tsf.functional.hflip(mask)
        
        if random.random() > 0.5:
            img = tsf.functional.vflip(img)
            mask = tsf.functional.vflip(mask)
        
        return img, mask

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index], cv2.IMREAD_COLOR)
        img = to_tensor(img)
        img = self.resize(img)
        
        mask = cv2.imread(self.mask_list[index], cv2.IMREAD_GRAYSCALE)
        mask = to_tensor(mask)
        mask = self.resize(mask)
        
        if self.is_train:
            img, mask = self._random_affine(img, mask)
            img = self._color_jitter(img) 
        
        img = to_grayscale(img)
        
        return img, mask
        
    def __len__(self):
        return len(self.img_list)
    
    
class BeetleDataset(Dataset):
    """
    Required directory configuration:
        train/valid
        ├─ 0/
        ├─ 1/
         ⁝
        ├─ n/
        └─ unknown/
    """
    # augmentation
    noise_scale = 0.01
    angle = 90
    translate = [0.1, 0.1]
    scale = [0.8, 1.2]
    shear = 5
    brightness = 0.2
    contrast= 0.2
    
    def __init__(self, data_list, num_classes, config, is_train=False):
        self.data_list = data_list
        self.is_train = is_train
        self.n_cls = num_classes
        
        self.resize = tsf.Resize(config['input_size'][::-1])
        
        self.augment = tsf.Compose([
            tsf.RandomAffine(
                degrees = self.angle,
                translate = self.translate,
                scale = self.scale,
                shear = self.shear
            ),
            tsf.ColorJitter(
                brightness = self.brightness,
                contrast = self.contrast
            ),
            tsf.RandomHorizontalFlip(0.5),
            tsf.RandomVerticalFlip(0.5)
        ])
    
    def __getitem__(self, index:int): 
        img_path = self.data_list[index]
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = to_tensor(img)
        img = self.resize(img)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.is_train:
                img = self.augment(img)
                img += self.noise_scale*torch.randn_like(img)
        
        img = to_grayscale(img)
        
        # cook class
        img_class = int(os.path.basename(os.path.dirname(img_path)))
        tag = torch.zeros(self.n_cls)
        if img_class != -1:
            tag[img_class] = 1
        
        return img, tag

    def __len__(self):
        return len(self.data_list)