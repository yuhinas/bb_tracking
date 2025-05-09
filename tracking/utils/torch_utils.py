from torch import nn
import torchvision.transforms as tsf
import torch


to_tensor =  tsf.ToTensor()
to_grayscale = tsf.Grayscale(1)

def cook_input(img, size, device=None):
    """
    PIL Image or numpy image to resized grascale torch tensor in (1, 1, H, W).
    size: Tuple, (h, w)
    """
    img = to_tensor(img).to(device)
    img = tsf.Resize(size)(img)
    img = to_grayscale(img).unsqueeze(0)
    return img

def center_crop_4x3(image, height=None):
    """Crop an image to aspect ratio 4:3.
    Args
        image: Torch tensor or PIL image.
        height: Int heigh of output size. Adaptively cropping if is None.
    """
    if height is None:
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            w, h = image.size
        
        l = min(w/4, h/3)
        w, h = int(4*l), int(3*l)
            
    else:
        w, h = int(4*height/3), height
    
    return tsf.CenterCrop((h,w))(image)

def center_crop_4x3_numpy(image, height=None):
    img_h, img_w = image.shape[:2]
    if height is None:
        l = min(img_w/4, img_h/3)
        w, h = int(4*l), int(3*l)
    else:
        w, h = int(4*height/3), height
        
    dw, dh = (img_w-w)//2, (img_h-h)//2
    cropped = image[dh:dh+h, dw:dw+w]
    return cropped

def crop_array_images(frame, boxes):
    h, w = frame.shape[:2]
    crops = []
    for b in boxes:
        left = int(max(b[0], 0))
        top = int(max(b[1], 0))
        right = int(min(b[2], w))
        bottom = int(min(b[3], h))
        crops.append(frame[top:bottom, left:right])
    return crops
        
def is_on_mouse(boxes:list, area_map, threshold=0.1):
    """
    area_map: Area map array in (H, W) composed of 1 and 0.
    """
    on_mouse = []
    for box in boxes:
        x1, y1, x2, y2 = box
        box_area = (x2-x1) * (y2-y1)
        mouse_area = torch.sum(area_map[int(y1):int(y2), int(x1):int(x2)])
        on_mouse.append(bool(mouse_area/(box_area+1e-6) > threshold))
    return on_mouse

def miou(stack1, stack2):
    """
    Inputs: Mask stack in 3 dim torch tensor composed of 1 and 0.
    """
    inter = torch.logical_and(stack1, stack2)
    inter = torch.sum(inter, dim=(1,2))
    union = torch.logical_or(stack1, stack2)
    union = torch.sum(union, dim=(1,2))
    return inter / (union+1e-6)

def kaiming_init(unit):
    if isinstance(unit, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(unit.weight.data)
        if unit.bias is not None:
            torch.nn.init.zeros_(unit.bias)
            
