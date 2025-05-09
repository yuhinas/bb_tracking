from torchvision import ops
import torch.nn.functional as fun
from torch import nn
import torch


def decode_line(line):
        line = line.split()
        objs = [s.split(',') for s in line[1:]]
        path, labels, boxes = line[0], [], []
        for obj in objs:
            labels.append(int(obj[0]))
            boxes.append([float(b) for b in obj[1:]])
        return path, labels, boxes
    
def yolo_line_to_boxes(lines:list):
    boxes = []
    for l in lines:
        _, _, b = decode_line(l)
        boxes += b
    return torch.Tensor(boxes)

def convert_boxes(boxes, size):
    """
    Convert [cx, cy, w, h] to [left, top, right, bottom], and
    rescale boxes back to image size.
    
    boxes: Torch Tensor
    size: (w, h)
    """
    boxes = ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
    boxes[:, [0, 2]] = size[0]*boxes[:, [0, 2]]
    boxes[:, [1, 3]] = size[1]*boxes[:, [1, 3]]
    return boxes

def generate_anchors(boxes, k=3):
    """
    Get k anchors via kmeans with IOU.
    boxes: Torch Tensor in (N, 4), [[x, y, w, h], [x, y, w, h], ...].
    """
    boxes[:, 0:2] = 0.
    ndx = torch.randint(len(boxes), size=(k,))
    anchors = boxes[ndx]  # random init anchor
    anchors_ = torch.empty_like(anchors)  # for stop signal
    
    stop = False
    while not stop:
        iou_table = ops.box_iou(boxes, anchors)
        clusters = torch.argmax(iou_table, axis=1)
    
        for i in range(k):
            anchors_[i], _ = torch.median(boxes[clusters==i], 0)
        
        stop = torch.all(anchors_==anchors)
        anchors = anchors_
    return anchors[:, 2:]

def leaky_sigmoid(x, leaky_scale=0.2):
    return (1+leaky_scale) * torch.sigmoid(x) - leaky_scale/2


class HeadDecoder(nn.Module):
    """
    Decode YOLO head from (B, 5, H, W) to
    (B, n_anchors*H*W, 5)
    """
    
    def __init__(self, anchors):
        super(HeadDecoder, self).__init__()
        self.anchors = anchors
        self.device = anchors.device
        self.head_size = (0, 0, 0, 0)
        
    def _create_offset_grids(self):
        bs, _, head_h, head_w = self.head_size
        grid_x = torch.arange(0, head_w, device=self.device)
        grid_y = torch.arange(0, head_h, device=self.device)
        self.grid_y, self.grid_x = torch.meshgrid(grid_y, grid_x)

    def _create_anchor_maps(self):
        bs, _, head_h, head_w = self.head_size
        size = (bs, len(self.anchors), head_h, head_w)
        self.anchor_w = torch.empty(size, device=self.device)
        self.anchor_h = torch.empty(size, device=self.device)
        for i in range(len(self.anchors)):
            self.anchor_w[:, i, ...] = self.anchors[i, 0]
            self.anchor_h[:, i, ...] = self.anchors[i, 1]
        
    def forward(self, yolo_head, is_train=False):
        bs, _, h, w = yolo_head.shape
        
        inf = yolo_head.reshape(
            bs, len(self.anchors), -1, h, w
        ).permute(0, 1, 3, 4, 2)
        
        objs = torch.sigmoid(inf[..., 0])
        cx = leaky_sigmoid(inf[..., 1])
        cy = leaky_sigmoid(inf[..., 2])
        w_scale = inf[..., 3]
        h_scale = inf[..., 4]
        clss = torch.sigmoid(inf[..., 5:])

        if self.head_size != yolo_head.shape:
            self.head_size = yolo_head.shape
            self.n_cls = inf.shape[-1] - 5
            self._create_offset_grids()
            self._create_anchor_maps()

        boxes = torch.empty_like(inf[..., :4])
        inf[..., 1] = (cx + self.grid_x) / w
        inf[..., 2] = (cy + self.grid_y) / h
        inf[..., 3] = torch.exp(w_scale) * self.anchor_w
        inf[..., 4] = torch.exp(h_scale) * self.anchor_h
        boxes = inf[..., 1:5]
        
        if is_train:
            return boxes, objs, clss
        else:
            boxes = boxes.reshape(bs, -1, 4)
            cls_confs, cls_ids = torch.max(clss, dim=-1)
            cls_ids = cls_ids.reshape(bs, -1, 1)
            confs = objs.reshape(bs, -1, 1) * cls_confs.reshape(bs, -1, 1)
            return boxes, cls_ids, confs
    

class YOLOLoss(nn.Module):
    """
    anchors: Torch Tensor with device in (N, 2).
    tragets: (labels, boxes)
    """
    num_classes = 1
    ignore = 0.5
    class_mask = 0.9
    
    # loss lambda
    lambda_box = 0.05
    lambda_obj = 1.5
    lambda_noobj = 0.5
    lambda_cls = 0.5
    
    def __init__(self, anchors, ):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.device = anchors.device
        
        self.decode_head = HeadDecoder(anchors)
        self.pad_anchors = torch.cat(
            [torch.zeros_like(self.anchors), self.anchors], dim=1
        )
        
    def _assign_to_anchors(self, boxes):
        boxes = torch.cat(
            [torch.zeros_like(boxes[:, 2:4]), boxes[:, 2:4]], dim=1
        )
        iou_metrix = ops.box_iou(boxes, self.pad_anchors)
        anchor_ids = torch.argmax(iou_metrix, dim=1)
        return anchor_ids
        
    def _cook_targets(self, targets):
        """Encode targets to head.
        """
        bs, _, h, w = self.head_size
        
        # build targets
        t_obj = torch.zeros(
            bs, len(self.anchors), h, w, requires_grad=False, device=self.device
        )
        t_boxes = torch.empty(
            bs, len(self.anchors), h, w, 4, requires_grad=False, device=self.device
        )
        t_clss = torch.zeros(
            bs, len(self.anchors), h, w, self.num_classes, 
            requires_grad=False, device=self.device
        )

        for b, (labels, boxes) in enumerate(zip(*targets)):
            if len(boxes) == 0:
                continue
            
            # offset of center (x, y), get grid (i, j)
            assert torch.all(boxes < 1.), 'Box annotations shold not exceed 1.'
            gt_i = torch.floor(boxes[:, 0:1] * w).long()
            gt_j = torch.floor(boxes[:, 1:2] * h).long()
            
            anchor_ids = self._assign_to_anchors(boxes)
            for n, anc_id in enumerate(anchor_ids):
                i, j = gt_i[n], gt_j[n]

                t_obj[b, anc_id, j, i] = 1
                t_boxes[b, anc_id, j, i] = boxes[n]
                t_clss[b, anc_id, j, i, labels[n]] = 1
                
        return t_obj, t_boxes, t_clss
    
    def _cook_inputs(self, yolo_head, targets):
        bs, _, h, w = self.head_size
        infer_boxes, infer_objs, infer_clss = self.decode_head(yolo_head, is_train=True)
        
        noobj_mask = torch.ones(
            bs, len(self.anchors), h, w, requires_grad=False, device=self.device
        )  # inculding obj and noobj, then ignore iou matching grids
        
        # set not training grids
        boxes = infer_boxes.reshape(bs, -1, 4)
        for b, (labels, t_boxes) in enumerate(zip(*targets)):
            if len(t_boxes) == 0:
                continue
            inf = ops.box_convert(boxes[b], in_fmt='cxcywh', out_fmt='xyxy')
            tar = ops.box_convert(t_boxes, in_fmt='cxcywh', out_fmt='xyxy')
            
            iou_metrix = ops.box_iou(tar, inf)
            max_iou, _ = torch.max(iou_metrix, dim=0)
            max_iou = max_iou.view(noobj_mask.shape[1:])
            noobj_mask[b, max_iou > self.ignore] = 0
            
        return noobj_mask, infer_boxes, infer_objs, infer_clss
    
    def _ciou_loss(self, b1_xywh, b2_xywh):
        if not len(b1_xywh)*len(b2_xywh):
            return torch.zeros(1, device=self.device)
        
        b1_xyxy = ops.box_convert(b1_xywh, in_fmt='cxcywh', out_fmt='xyxy')
        b2_xyxy = ops.box_convert(b2_xywh, in_fmt='cxcywh', out_fmt='xyxy')

        # iou
        inter_min = torch.max(b1_xyxy[:, 0:2], b2_xyxy[:, 0:2])
        inter_max = torch.min(b1_xyxy[:, 2:4], b2_xyxy[:, 2:4])
        inter_wh = torch.max(inter_max-inter_min, torch.zeros_like(inter_min))
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        b1_area = b1_xywh[:, 2] * b1_xywh[:, 3]
        b2_area = b2_xywh[:, 2] * b2_xywh[:, 3]
        union_area = b1_area + b2_area - inter_area

        iou = inter_area / torch.clamp(union_area, min=1e-6)

        # distance
#         d = torch.sum(torch.pow(b1_xywh[:, 0:2]-b2_xywh[:, 0:2], 2), dim=1)
        d = torch.sum(torch.abs(b1_xywh[:, 0:2]-b2_xywh[:, 0:2]), dim=1)
        enclose_min = torch.min(b1_xyxy[:, 0:2], b2_xyxy[:, 0:2])
        enclose_max = torch.max(b1_xyxy[:, 2:4], b2_xyxy[:, 2:4])
        enclose_wh = enclose_max - enclose_min
#         c = torch.sum(torch.pow(enclose_wh, 2), dim=1)
        c = torch.sum(torch.abs(enclose_wh), dim=1)

        dis = d / torch.clamp(c, min=1e-6)

        # angle
        b1_atan = torch.atan(b1_xywh[:, 2]/torch.clamp(b1_xywh[:, 3], min=1e-6))
        b2_atan = torch.atan(b2_xywh[:, 2]/torch.clamp(b2_xywh[:, 3], min=1e-6))
#         v = (4/3.1415926**2) * torch.pow(b1_atan-b2_atan, 2)
        v = (4/3.1415926**2) * torch.abs(b1_atan-b2_atan)
        a = v / torch.clamp((1. - iou + v), min=1e-6)

        # loss
        ciou_loss = 1. - iou + dis + a*v
        return ciou_loss
            
    def forward(self, yolo_head, targets):
        self.head_size = yolo_head.shape
        bs, _, h, w = self.head_size
        
        # encode target
        t_obj, t_boxes, t_clss = self._cook_targets(targets)
        noobj_mask, infer_boxes, infer_objs, infer_clss = self._cook_inputs(yolo_head, targets)

        # ciou loss
        ciou_loss = self._ciou_loss(
            infer_boxes[t_obj.bool()], t_boxes[t_obj.bool()]
        )
        box_loss = self.lambda_box * torch.mean(ciou_loss)

        # obj loss
        obj_loss = fun.binary_cross_entropy(
            infer_objs[t_obj.bool()], t_obj[t_obj.bool()], reduction='sum'
        )
        noobj_loss = fun.binary_cross_entropy(
            infer_objs[noobj_mask.bool()], t_obj[noobj_mask.bool()], reduction='sum'
        )
        
        obj_losses = (self.lambda_obj*obj_loss + self.lambda_noobj*noobj_loss)
        
        # class loss
        cls_loss = fun.binary_cross_entropy(
            infer_clss[t_obj.bool()], t_clss[t_obj.bool()], reduction='mean'
        )
        
        grid_loss =  obj_losses/torch.sum(t_obj+noobj_mask) + self.lambda_cls*cls_loss
        return box_loss, grid_loss