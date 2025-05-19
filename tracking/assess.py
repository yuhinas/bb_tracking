from PIL import Image
from tqdm import tqdm
from glob import glob
from torchvision import ops
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import csv
import cv2
import sys
import os

from utils import app, utils, torch_utils, yolo_utils

# ---
# python assess.py [model_type] [config.yaml] [optional_comment]
#
# [model_type]: segmenter, classifier, detector
# [optional_comment]: Optional comment text that will be appended to the output file names for distinction.
# ---

if len(sys.argv) == 1:
    print('Please assign the model to assess.')
    sys.exit()
elif len(sys.argv) == 2:
    print('Please give the config path for the model.')
    sys.exit()
    
if len(sys.argv) >= 4:
    comment = '-' + '-'.join(sys.argv[3:])
else:
    comment = ''

# load config
with open(sys.argv[2]) as yf:
    configs = yaml.safe_load(yf)
    
# set device
if configs['device'] == 'cpu':
    device = 'cpu'
else:
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs['device'])
    
    
if sys.argv[1] == 'segmenter':
    config = configs['Segmenter']
    
    img_list = utils.read_txt_to_list(config['testing_data_list'])
    model = app.Segmenter(config, device)
    
    # match
    mious, paths = [], []
    i = 0
    for path in tqdm(img_list, ncols=80):
        gt_path = path.replace('/img/', '/mask/')
        gt_mask = Image.open(gt_path)
        gt_mask = torch_utils.cook_input(gt_mask, config['input_size']).squeeze()

        # pass empty gt
        if torch.any(gt_mask==1)==False:
            continue

        if i%config['testing_batch_size'] == 0:
            batch_imgs, batch_masks = [], []

        batch_imgs.append(Image.open(path))
        batch_masks.append(gt_mask)

        if (i+1)%config['testing_batch_size']==0 or i+1==len(img_list):
            pd_masks = model.cut_batch(batch_imgs)
            gt_masks = torch.stack(batch_masks)

            mious += torch_utils.miou(pd_masks, gt_masks).tolist()
            
        paths.append(path)
        i += 1

    file_name = f'logs/miou{comment}'
    # write assement
    with open(f'{file_name}.csv', 'w') as cf:
        write = csv.writer(cf)
        for p, i in zip(paths, mious):
            write.writerow([p, i])
    
    # plot
    plt.figure(dpi=150)
    plt.xlabel('MIOU')
    plt.ylabel('Number')
    plt.grid(alpha=0.6)
    plt.hist(mious, bins=20, range=(0, 1), alpha=0.8)
    
    plt.savefig(f'{file_name}.png')
    print(f'Save the results as {file_name}.')
    plt.show()
    
    
elif sys.argv[1] == 'classifier':
    config = configs['Classifier']
    
    folder_list = utils.read_txt_to_list(config['testing_data_folder'])
    img_list = []
    for d in folder_list:
        img_list += sorted(glob(f'{d}/*/*[.jpg, .png]'))
    
    n_cls = len(configs['classes'])
    model = app.Classifier(n_cls, config, device)

    c_matrix = np.zeros((n_cls+1, n_cls+1))
    corrects = []

    i = 0
    for path in tqdm(img_list, ncols=80):
        i += 1
        if i%config['testing_batch_size'] == 1:
            batch_imgs, batch_gts = [], []

        gt = int(os.path.basename(os.path.dirname(path)))
        batch_gts.append(gt)
        batch_imgs.append(Image.open(path))

        if i%config['testing_batch_size']==0 or i==len(img_list):
            _ = model.mark_batch(batch_imgs)
            batch_pds = model.id

            for pd, gt in zip(batch_pds, batch_gts):
                c_matrix[gt+1, pd+1] += 1
                corrects.append(pd==gt)
        
    acc = sum(corrects) / len(corrects)
    # normalize confusion matrix
    numbers = np.sum(c_matrix, axis=1, keepdims=True)
    cm = c_matrix / numbers  # matrix in percentage
    labels = [i for i in range(-1, n_cls)]

    # set map
    def one_gaussian2d(cx, cy, a, size, width):
        x, y = np.meshgrid(range(size), range(size))
        return a*np.exp(-2*((x-cx)**2+((y-cy)**2))/width**2)

    cm_color = cm**(1/6)
    size = len(cm)
    z = np.zeros((100*size, 100*size))
    for i in range(size):
        for j in range(size):
            z += one_gaussian2d(50+100*j, 50+100*i, cm_color[i, j], 100*size, 40*np.log(size))


    # set figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xticks([50+100*i for i in range(size)])
    ax.set_yticks([50+100*i for i in range(size)])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Inference')
    ax.set_ylabel('Truth (Normalized)')
    ax.set_title(f'Accuracy: {acc:.2f}')

    plt.imshow(z, cmap='Blues')  # here to set cmap

    # set text
    for i in range(size):
        for j in range(size):
            c = 'k' if round(cm[i, j],2) < 0.25 else 'w'
            ax.text(
                50+100*j, 50+100*i, round(cm[i, j],2), 
                ha="center", va="center", color=c
            )
    
    path = f'logs/confusion-matrix{comment}.png'
    plt.savefig(path)
    print(f'Saved the result as {path}.')
    plt.show()
    
    
elif sys.argv[1] == 'detector':
    config = configs['Detector']
    model = app.Detector(config, device)
    
    # get gt data
    data = {}
    with open(config['testing_data_list']) as tf:
        for line in tf:
            path, _, boxes = yolo_utils.decode_line(line)
            data[path] = boxes

    results = []  # [conf, answer]
    total_gt = 0

    # detect and match
    total_n = len(data)
    with tqdm(desc='Detecting', total=total_n, ncols=60) as pbar:
        for n, (path, gt_boxes) in enumerate(data.items()):
            init_batch = n%config['testing_batch_size']==0
            full_batch = (n+1)%config['testing_batch_size']==0

            if init_batch:
                batch_imgs, batch_gts, img_sizes = [], [], []

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            batch_imgs.append(img)
            batch_gts.append(gt_boxes)
            img_sizes.append(img.shape[:2])
            pbar.update(1)
    
            if full_batch:
                batch_pds, batch_scores = model.detect_batch(batch_imgs)
                
                for bs in range(config['testing_batch_size']):
                    total_gt += len(gt_boxes)
                    gt = torch.Tensor(batch_gts[bs]).to(device)
                    gt = ops.box_convert(gt, in_fmt='cxcywh', out_fmt='xyxy')
                    
                    # recover size
                    gt[:, [0, 2]] *= img_sizes[bs][1]
                    gt[:, [1, 3]] *= img_sizes[bs][0]
                    
                    pd = batch_pds[bs]
                    iou_matrix = ops.box_iou(gt, pd)
                    corrects = torch.any(iou_matrix>config['testing_iou_threshold'], dim=0)
                    for s, c in zip(batch_scores[bs].tolist(), corrects):
                        results.append([s, bool(c)])


    # sort results and get precision and recall
    results = sorted(results, key=lambda x: x[0], reverse=True)
    corrects = [r[1] for r in results]

    precision, recall = [], []
    for i in tqdm(range(len(results)), desc='Counting', ncols=60):
        pc = sum(corrects[0:i+1]) / len(corrects[0:i+1])
        precision.append(pc)

        rc = sum(corrects[0:i+1]) / total_gt
        recall.append(rc)

    # plot

    plt.figure(dpi=150)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Precision - Recall')
    plt.grid(alpha=0.6)
    plt.plot(recall, precision)
    
    path = f'logs/precision-recall{comment}.png'
    plt.savefig(path)
    print(f'Saved the result as {path}.')
    plt.show()