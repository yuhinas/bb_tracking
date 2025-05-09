from torchvision import ops
from scipy.optimize import linear_sum_assignment
from scipy.linalg import inv
import torchvision.transforms as tsf
import numpy as np
import torch

from nets import classifier, segmenter, detector
from utils import torch_utils, yolo_utils


def compute_iou(b1, b2):
    """
    Return IoU matrix of b1 (N, 4) and b2 (M, 4) in (N, M)
    """
    a1 = np.abs((b1[:, 2]-b1[:, 0]) * (b1[:, 3]-b1[:, 1]))
    a2 = np.abs((b2[:, 2]-b2[:, 0]) * (b2[:, 3]-b2[:, 1]))

    lt = np.maximum(b1[:, np.newaxis, :2], b2[:, :2])
    rb = np.minimum(b1[:, np.newaxis, 2:], b2[:, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)

    inter = wh[..., 0] * wh[..., 1]
    union = a1[:, np.newaxis] + a2 - inter
    iou = inter / (union+1e-6)
    return iou

class Segmenter:
    
    def __init__(self, config, device='cuda'):
        self.device = device
        self.input_size = config['input_size'][::-1]
        
        if config['model'] == 'default':
            self.model = segmenter.UNet()
        elif config['model'] == 'official':
            self.model = segmenter.official_model() 
            
        state_dict = torch.load(config['state_dict_path'])
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval()
        
        print(f'Segmenter {config["state_dict_path"]} is ready on "{device}"!')
    
    def cut(self, numpy_image, keep_size=True):
        h, w, _= numpy_image.shape
        img = torch_utils.cook_input(numpy_image, self.input_size, self.device)
        with torch.no_grad():
            output = self.model(img)
            if keep_size:
                output = tsf.Resize((h, w))(output)
            self.mask = torch.where(output>0, 1., 0.).squeeze()
        return self.mask
        
    def cut_batch(self, numpy_images:list):
        ts_imgs = []
        for img in numpy_images:
            img = torch_utils.cook_input(img, self.input_size, self.device)
            ts_imgs.append(img)
        ts_imgs = torch.cat(ts_imgs, dim=0)
       
        with torch.no_grad():
            output = self.model(ts_imgs).squeeze()
            self.mask = torch.where(output>0, 1., 0.)
        return self.mask
    

class Classifier:
    
    def __init__(self, n_classes, config, device='cuda'):
        self.device = device
        self.input_size = config['input_size']
        self.thrs = config['threshold']
        
        # model
        if config['model'] == 'default':
            self.model = classifier.ResNet(n_classes)
        else:
            self.model = classifier.official_model(config['model'], n_classes)
            
        state_dict = torch.load(config['state_dict_path'])
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()
        
        print(f'Classifier {config["state_dict_path"]} is ready on "{self.device}"!')
        
    def _cook_results(self, output):
        """
        Assign to "unknown" if 
            (1) all classes under the threshold, or
            (2) more than two classes larger than the threshold.
        """
        score, cls_id = torch.max(output, dim=-1)
        
#         mask = score<self.thrs
        mask = torch.sum(output>self.thrs, dim=1)!=1

        score[mask], cls_id[mask] = 0, -1
        score, cls_id = score.tolist(), cls_id.tolist()
        return cls_id, score
        
    def mark(self, numpy_image):
        img = torch_utils.cook_input(numpy_image, self.input_size, self.device)
        with torch.no_grad():
            output = self.model(img)
            self.raw = output
            i, s = self._cook_results(output)
            self.id, self.score = *i, *s
        return self.id, self.score
    
    def mark_batch(self, numpy_images):
        ts_imgs = []
        for img in numpy_images:
            img = torch_utils.cook_input(img, self.input_size, self.device)
            ts_imgs.append(img)
        ts_imgs = torch.cat(ts_imgs, dim=0)

        with torch.no_grad():
            output = self.model(ts_imgs)
            self.raw = output
            self.id, self.score = self._cook_results(output)
        return self.id, self.score
    
    
class Detector:
    
    def __init__(self, config, device='cuda'):
        self.device = device
        self.input_size = config['input_size'][::-1]
        self.s_thrs = config['score_threshold']
        self.b_thrs = config['nms_iou_threshold']
        
        checkpoint = torch.load(config['state_dict_path'], map_location=device)
        self.model = detector.YOLOv4(len(checkpoint['anchors'])).to(device).eval()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.decode_head = yolo_utils.HeadDecoder(checkpoint['anchors'])
        
        print(f'Detecor {config["state_dict_path"]} is ready on "{device}"!')
        
    def detect(self, numpy_image):
        h, w = numpy_image.shape[:2]
        img = torch_utils.cook_input(numpy_image, self.input_size, self.device)

        with torch.no_grad():
            yolo_head = self.model(img)
            boxes, _, scores = self.decode_head(yolo_head, is_train=False)
            boxes, scores = boxes.squeeze(0), scores.squeeze()
            
            boxes = boxes[scores>self.s_thrs]
            scores = scores[scores>self.s_thrs]
            
            boxes = ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * h
            
            keep = ops.nms(boxes, scores, self.b_thrs)
            self.boxes, self.scores = boxes[keep], scores[keep]
        return self.boxes, self.scores
            
    def detect_batch(self, numpy_images):
        ts_imgs, self.boxes, self.scores = [], [], []
        for img in numpy_images:
            img = torch_utils.cook_input(img, self.input_size, self.device)
            ts_imgs.append(img)
        ts_imgs = torch.cat(ts_imgs, dim=0)
        
        with torch.no_grad():
            yolo_head = self.model(ts_imgs)
            boxes, _, scores = self.decode_head(yolo_head)
                        
        for bs in range(len(numpy_images)):
            h, w = numpy_images[bs].shape[:2]
            b, s = boxes[bs].squeeze(0), scores[bs].squeeze()
            
            b, s = b[s>self.s_thrs], s[s>self.s_thrs]
            b = ops.box_convert(b, in_fmt='cxcywh', out_fmt='xyxy')
            b[:, [0, 2]] = b[:, [0, 2]] * w
            b[:, [1, 3]] = b[:, [1, 3]] * h
            keep = ops.nms(b, s, self.b_thrs)
            
            self.boxes.append(b[keep])
            self.scores.append(s[keep])
        return self.boxes, self.scores
    
    
class KalmanFilter:
    """
    measurement (z):
    (xmin, ymin, xmax, ymax)
    
    state (x):
    (xmin, ymin, xmax, ymax, vxmin, vymin, vxmax, vymax)
    """
    # state transition function
    F = np.array([[1., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1.]])
    # measurement function
    H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0.]])

    """Covariance
    Considering the variances will vary according to box size,
    the prior R shold be scaled by (w+h)/2 in advance, name scaled R.
    """
    # initial state covariance
    scaled_P = np.diag(
        [0.01, 0.01, 0.01, 0.01, 0.004, 0.004, 0.004, 0.004]
    )
    # process covariance
    scaled_Q = np.diag(
        [0.025, 0.025, 0.0025, 0.025, 0.0004, 0.0004, 0.0004, 0.0004]
    )
    # measurement covariance
    scaled_R = np.diag(
        [0.01, 0.01, 0.01, 0.01]
    )
    
    def initiate(self, z):
        self.x = np.zeros(8)
        self.x[:4] = z
        
        obj_size = (np.sum(z[2:4])/2)**2
        self.P = self.scaled_P * obj_size
        self.R = self.scaled_R * obj_size
        self.Q = self.scaled_Q * obj_size
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.prior = self.H @ self.x
        return self.prior
    
    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        
        y = z - self.prior
        self.x += K @ y
        self.P = self.P - K @ self.H @ self.P
        
        self.posterior = self.H @ self.x
        return self.posterior
    

class IDTracker:
    
    def __init__(self, config):
        self.iou_thrs = config['iou_threshold']
        self.kill_thrs = config['kill_threshold']
        self.class_thrs = config['class_threshold']
        
        # initialization
        self.reset()
    
    def reset(self):
        """
        Reset all tracking parameter and
        clean trace
        """
        self.states = {
            'id': np.array([], dtype=int), 
            'box': np.array([]),
            'absent': np.array([], dtype=int), 
            'filter': [], 
            'iou': np.array([], dtype=float),
        }
        self.id = 0
        self.trace = []
        
    def _update_trace(self):
        self.trace.append({
            'Track IDs': self.states['id'].tolist(),
            'Boxes': self.states['box'].tolist(),
            'IoU Scores': self.states['iou'].tolist()
        })
    
    def _remove_items(self, data:list, to_keep):
        data = np.array(data)
        data = data[to_keep]
        return data.tolist()
             
    def _add_online(self, boxes):
        self.states['id'] = np.concatenate(
            [self.states['id'], np.arange(len(boxes))+self.id]
        )
        self.states['absent'] = np.concatenate(
            [self.states['absent'], np.zeros(len(boxes))]
        )
        self.states['iou'] = np.concatenate(
            [self.states['iou'], np.ones(len(boxes))]
        )
        self.id += len(boxes)

        if len(self.states['box']) == 0:
            self.states['box'] = boxes
        else:
            self.states['box'] = np.concatenate([self.states['box'], boxes])
        
        for i in range(len(boxes)):
            kf = KalmanFilter()
            kf.initiate(boxes[i])
            self.states['filter'].append(kf)
    
    def _kill_online(self, to_keep:bool):
        self.states['id'] = self.states['id'][to_keep]
        self.states['box'] = self.states['box'][to_keep]
        self.states['iou'] = self.states['iou'][to_keep]
        self.states['absent'] = self.states['absent'][to_keep]
        
        new_kfs = []
        for kf, keep in zip(self.states['filter'], to_keep):
            if keep:
                new_kfs.append(kf)
        self.states['filter'] = new_kfs
            
    def _kf_predict(self):
        for i, kf in enumerate(self.states['filter']):
            prior = kf.predict()
            self.states['box'][i] = prior
            
    def _pick_up_unmatched(self, matching_state, success_ids):
        ndx = 0
        for i, b in enumerate(matching_state):
            if b==True and ndx not in success_ids:
                matching_state[i] = False
            elif b == False:
                ndx += 1
            ndx += 1
        return matching_state
    
    def track(self, boxes:np.ndarray):
        """
        Input
        boxes: Numpy array in (N, 4)
        
        Return
        List of track IDs
        """
        
        if len(boxes) != 0:
            
            if len(self.states['id']) == 0:
                self._add_online(boxes)
                
            else:
                self._kf_predict()
                iou = compute_iou(boxes, self.states['box'])
                
                box_matched = np.any(iou>self.iou_thrs, axis=1)
                box_matched_i = np.where(box_matched)[0]
                track_matched = np.any(iou>self.iou_thrs, axis=0)
                track_matched_i = np.where(track_matched)[0]
                to_kill_items = []
                
                # matching
                if np.any(box_matched) and np.any(track_matched):
                    iou = iou[box_matched][:, track_matched]
                    measurement_i, prior_i = linear_sum_assignment(iou, maximize=True)
                    
                    matched_box = boxes[box_matched]
                    # matched boxes: update the information
                    for m, p in zip(measurement_i, prior_i):
                        z = matched_box[m]
                        posterior = self.states['filter'][track_matched_i[p]].update(z)
                        self.states['box'][track_matched_i[p]] = posterior
                        self.states['iou'][track_matched_i[p]] = round(iou[m, p], 2)
                        
                        # average class code
                        track_id = self.states['id'][track_matched_i][p]
                        
                    # reset absent
                    self.states['absent'][track_matched_i] = 0
                    
                    # pick unmatched boxes or tracks
                    if np.sum(box_matched) > np.sum(track_matched):
                        box_matched = self._pick_up_unmatched(box_matched, measurement_i)

                    elif np.sum(box_matched) < np.sum(track_matched):
                        track_matched = self._pick_up_unmatched(track_matched, prior_i)
            
                # missing tracks: use kf predictions or kill it
                absent = track_matched==False
                self.states['absent'][absent] += 1
                self.states['iou'][absent] = 0.
                
                to_keep = self.states['absent'] < self.kill_thrs
                self._kill_online(to_keep=to_keep)
                        
                # unmatched boxes: assign new IDs
                self._add_online(boxes[box_matched==False])
        
        elif len(self.states['id']) != 0:
            self.states['absent'] += 1
            self.states['iou'][:] = 0.

            to_keep = self.states['absent'] < self.kill_thrs
            self._kill_online(to_keep=to_keep)

            self._kf_predict()
            
        # detele boxes with negtive w and h
        if len(self.states['box']) != 0:
            w = self.states['box'][:, 2] - self.states['box'][:, 0]
            h = self.states['box'][:, 3] - self.states['box'][:, 1]
            is_boxes = np.logical_and(w>0, h>0)
            self._kill_online(to_keep=is_boxes)
        
        self._update_trace()


class Tracker:
    
    def __init__(self, classes:dict, config):
        self.class_names = classes
        self.num_classes = len(classes)
        self.match_thrs = config['match_threshold']
        self.w = config['weights']
        
        # init
        self.reset()
        
    def reset(self):
        self.trace = []
        self.states = {
            'online': np.array([False]*self.num_classes),
            'box': np.empty((self.num_classes, 4)),
            'filter': [KalmanFilter() for _ in range(self.num_classes)]
        }
    
    def track(self, boxes, codes):
        """
        Args:
            boxes: Numpy array (N, 4) of bounding boxes in format:
                   [left, top, right, bottom].
            codes: Numpy array (N, n_classes) of class codes.
        """
        iou_scores = np.zeros(self.num_classes)
        cls_scores = np.zeros(self.num_classes)
        cls_id = []
        
        
        if len(boxes)!=0:
            score_matrix = np.copy(codes)
            iou_backup = np.zeros_like(score_matrix)
            
            any_online = any(self.states['online'])
            if any_online:
                    
                online = self.states['online']
                priors = self.states['box'][online]

                ious = compute_iou(boxes, priors)
                score_matrix[:, online] = self.w[0]*codes[:, online] + self.w[1]*ious
                iou_backup[:, online] = ious
            
            seq_id, cls_id = linear_sum_assignment(score_matrix, maximize=True)
            good = score_matrix[seq_id, cls_id] > self.match_thrs
            seq_id, cls_id = seq_id[good], cls_id[good]
            
            # kalman predict
            for i in np.where(self.states['online'])[0]:
                self.states['filter'][i].predict()
            
            pre_online = np.copy(self.states['online'])
            self.states['online'][:] = False
            for s, c in zip(seq_id, cls_id):               
                cls_scores[c] = codes[s, c]
                iou_scores[c] = iou_backup[s, c]
                self.states['online'][c] = True

                if pre_online[c]:
                    self.states['box'][c] = self.states['filter'][c].update(boxes[s])
                else:
                    self.states['filter'][c].initiate(boxes[s])
                    self.states['box'][c] = boxes[s]
                
        else:
            self.states['online'][:] = False
            
        # update trace
        label_ids = np.where(self.states['online'])[0].tolist()
        footprints = {
            'Label IDs': label_ids,
            'Labels': [self.class_names[i] for i in label_ids],
            'Boxes': self.states['box'][label_ids].tolist(),
            'Label Scores': cls_scores[label_ids].tolist(),
            'IoU Scores': iou_scores[label_ids].tolist()
        }
        self.trace.append(footprints)

