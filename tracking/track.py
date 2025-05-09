from tqdm import tqdm
import numpy as np
import json
import yaml
import cv2
import sys
import os
import torch
import csv
import re
from utils import app, torch_utils, utils

# ---
# python track-HSP.py config.yml
# ---

if len(sys.argv) == 1:
    print('Please give the config path for the tracking mission.')
    sys.exit()

def get_time(path):
    """
    For file name in format '[Chanel] date %H.%M.%S'.
    """
    t_regex = re.compile(r'\d\d\.\d\d\.\d\d')
    t = t_regex.search(path).group()
    t = list(map(int, t.split('.')))
    return t[0]*3600 + t[1]*60 + t[2]

def frame2time(f_no: np.array, fps, init_time):
    second = f_no / fps
    second = second + init_time
    second = second % 86400  # align 24 hours
    h = (second // 3600).astype(int)
    m = (second % 3600 // 60).astype(int)
    s = (second % 60).astype(int)
    t = [f'{i:0>2d}:{j:0>2d}:{k:0>2d}' for i, j, k in zip(h, m, s)]
    return t

# load configs
with open(sys.argv[1]) as yf:
    configs = yaml.safe_load(yf)

assert configs['mode'] in ['class', 'id']
label_dict = configs['classes']
batch_size = configs['batch_size']

if configs['device'] == 'cpu':
    device = 'cpu'
else:
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs['device'])
#print(torch.zeros(1).cuda())
# load models
detector = app.Detector(configs['Detector'], device)
segmenter = app.Segmenter(configs['Segmenter'], device)

if configs['mode'] == 'class':
    classifier = app.Classifier(len(label_dict), configs['Classifier'], device)
    tracker = app.Tracker(label_dict, configs['Tracker'])
elif configs['mode'] == 'id':
    tracker = app.IDTracker(configs['IDTracker'])

# load video groups
with open(configs['video_dict']) as jf:
    video_dict = json.load(jf)

total_vids = 0
for v in video_dict.values():
    total_vids += len(v)

# Create the log file for videos to check
log_file = f'{configs["output_dir"]}/videos_to_check-track.txt'
if not os.path.exists(log_file):
    with open(log_file, 'w') as lf:
        lf.write("Please confirm whether the output (raw-tracks, on-mouse) of the following videos is complete:\n")
    
total_t, n_vid = 0, 0
# for each group
for exp, v_list in video_dict.items():
    
    # create output dir
    out_dir = f'{configs["output_dir"]}/raw-tracks/{exp}'
    os.makedirs(out_dir, exist_ok=True)
    out_mouse_dir = f'{configs["output_dir"]}/mouse-record/{exp}'
    os.makedirs(out_mouse_dir, exist_ok=True)
    
    tracker.reset()
    
    for v_path in v_list:
        vid = cv2.VideoCapture(v_path)
        vid_name = os.path.basename(v_path)
        
        csv_name = f'{os.path.splitext(vid_name)[0]}.csv'
        csv_path = f'{out_mouse_dir}/{csv_name}'

        total = int(vid.get(7))
        n_vid += 1
        title = f'[{n_vid}/{total_vids}] {exp}: {vid_name}'
        
        if configs['center_crop_4x3']:
            w, h = int(vid.get(4)*4/3), int(vid.get(4))
        else:
            w, h = int(vid.get(3)), int(vid.get(4))
        
        data = {}
        data['Video'] = {
            'Path': v_path,
            'Crop 4x3': configs['center_crop_4x3'],
            'Width': w,
            'Height': h,
            'Length': total,
            'FPS': vid.get(5)
        }
        json_name = f'{out_dir}/{os.path.splitext(vid_name)[0]}.json'
        init_time = get_time(json_name)
        # if json is exist, pass para
        if os.path.exists(json_name):
            print(f'{json_name} is exist, skip.')
            continue
        
        on_mouse_record = []
        
        with open(csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['time', 'area_pixels', 'centroid_x', 'centroid_y'])

            with tqdm(desc=title, total=total, ncols=100, leave=False) as pbar:
                fail_count = 0
                max_failures = 2700

                while vid.isOpened():
                    f_no = int(vid.get(1))
                    success, frame = vid.read()

                    if f_no + 1 >= total:
                        break
                        
                    if not success:
                        with open(log_file, 'a') as lf:
                            lf.write(f"{v_path} - Read failed at frame {f_no}\n")

                        fail_count += 1

                        if fail_count >= max_failures:
                            skip_message = f"Skipping {v_path} due to consecutive failures at frame {f_no}."
                            with open(log_file, 'a') as lf:
                                lf.write(skip_message + "\n")
                            print(skip_message)
                            break
                        
                        continue

                    fail_count = 0

                    # crop
                    if configs['center_crop_4x3']:
                        frame = torch_utils.center_crop_4x3_numpy(frame, h)

                    init_batch = f_no%batch_size==0
                    full_batch = (f_no+1)%batch_size==0

                    # segment
                    if f_no%configs['mouse_frequency'] == 0:
                        mouse_area = segmenter.cut(frame, keep_size=True)

                        # Calculate area and center
                        mouse_np = mouse_area.cpu().numpy()
                        y_indices, x_indices = np.where(mouse_np > 0)  # Find all non-zero pixels
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            centroid_x = round(np.mean(x_indices), 6)
                            centroid_y = round(np.mean(y_indices), 6)
                            mouse_true_area = len(y_indices)  # Number of foreground pixels
                        else:
                            centroid_x = None
                            centroid_y = None
                            mouse_true_area = 0

                        # Calculate time for current frame
                        frame_time = frame2time(np.array([f_no]), data['Video']['FPS'], init_time)[0]

                        # Write to CSV
                        csv_writer.writerow([
                            frame_time,
                            mouse_true_area,
                            centroid_x,
                            centroid_y
                        ])

                    if init_batch:
                        batch_frames = []

                    batch_frames.append(frame)

                    if full_batch or (f_no+1)==total:
                        # detect
                        batch_boxes, _ = detector.detect_batch(batch_frames)

                        for i in range(len(batch_frames)):
                            boxes = batch_boxes[i].cpu().numpy()

                            if configs['mode'] == 'class':
                                class_codes = []
                                if len(boxes) > 0:
                                    crops = torch_utils.crop_array_images(frame, boxes)

                                    # classify
                                    _ = classifier.mark_batch(crops)
                                    class_codes = classifier.raw.cpu().numpy()

                                # track
                                tracker.track(boxes, class_codes)

                            elif configs['mode'] == 'id':
                                tracker.track(boxes)

                            # on mouse
                            boxes =tracker.trace[-1]['Boxes']
                            on_mouse = torch_utils.is_on_mouse(boxes, mouse_area, threshold=0.1)
                            tracker.trace[-1]['On Mouse'] = on_mouse

                        if full_batch:
                            pbar.update(batch_size)
                        else:
                            pbar.update(total%batch_size)

                data['Tracks'] = tracker.trace

                json_name = f'{out_dir}/{os.path.splitext(vid_name)[0]}.json'
                json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
                with open(json_name, 'w') as jf:
                    json.dump(data, jf, indent=4)

                tracker.trace = []

                t = pbar.format_dict['elapsed']
                total_t += t

            print(f'{title} finished. Excution time: {tqdm.format_interval(t)}')
            
print(f'Tracking Completed. Total excution time: {tqdm.format_interval(total_t)}')