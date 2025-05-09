from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import csv
import re
import os

RAW_TRACK_DIR = '../../output/raw-tracks'
OUTPUT_DIR = '../../output/on-mouse'

START_HOUR = 20
END_HOUR = 5

range_start = START_HOUR 
range_end = END_HOUR + 24

# function
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

fields = ['Time', 'H', 'O', 'X', 'nn', 'ss', 'xx']
fields_2 = ['', 'H', 'O', 'X', 'nn', 'ss', 'xx', 'total_frame']
fields_3 = ['Hour', 'H', 'O', 'X', 'nn', 'ss', 'xx', 'total_frame', 'total_sec']
markers = ['H', 'O', 'X', 'nn', 'ss', 'xx']
colors = ['r', 'g', 'b', 'c', 'm', 'y']
sex = ['M', 'F', 'F', 'F', 'M', 'M']

for exp_dir in tqdm(sorted(glob(f'{RAW_TRACK_DIR}/*')), desc='Generating states and statistic'):

    states = []
    counter = np.zeros((2, 6))
    total_length = 0

    exp_no = os.path.basename(exp_dir)
    track_list = sorted(glob(f'{exp_dir}/*.json'))

    for path in track_list:

        with open(path) as jf:
            data = json.load(jf)

        vid_info = data['Video']
        tracks = data['Tracks']
        fps = vid_info['FPS']
        length = vid_info['Length']
        total_length += length

        ### get time
        init_time = get_time(path)

        f_nos = np.arange(0, length)
        time_points = frame2time(f_nos, fps, init_time)

        for f, t in enumerate(tracks):
            s = [0]*7
            try:
                s[0] = time_points[f]
            except:
                print(f'{path} has exception')
                break
            for mark, on_mouse in zip(t['Labels'], t['On Mouse']):
                ndx = fields.index(mark)

                if on_mouse:
                    s[ndx] = 2
                    counter[0, ndx-1] += 1
                    counter[1, ndx-1] += 1
                else:
                    counter[0, ndx-1] += 1
                    s[ndx] = 1

            states.append(s)

    if not os.path.exists(f'{OUTPUT_DIR}/{exp_no}'):
        os.makedirs(f'{OUTPUT_DIR}/{exp_no}')

    # save states
    csv_path = f'{OUTPUT_DIR}/{exp_no}/states.csv'
    with open(csv_path, 'w') as cf:
        write = csv.writer(cf)
        write.writerow(fields)
        write.writerows(states)

    # save statistic
    counter = counter / total_length
    total_frames = total_length
    sta = np.round(counter, 6).tolist()
    sta[0] = ['Detected / Total Frame'] + sta[0] + [total_frames]
    sta[1] = ['On Mouse / Total Frame'] + sta[1] + [total_frames]

    csv_path = f'{OUTPUT_DIR}/{exp_no}/statistic.csv'
    with open(csv_path, 'w') as cf:
        write = csv.writer(cf)
        write.writerow(fields_2)
        write.writerows(sta)

data_list = glob(f'{OUTPUT_DIR}/*/states.csv')

for csv_path in tqdm(data_list, desc='Hourly data output'):
    exp_no = os.path.basename(os.path.dirname(csv_path))

    total_frames = {f'{n%24:0>2d}': 0 for n in range(range_start, range_end)}
    total_seconds = {f'{n%24:0>2d}': 0 for n in range(range_start, range_end)}
    counter_detected = {f'{n%24:0>2d}': np.zeros(6) for n in range(range_start, range_end)}
    counter_on_mouse = {f'{n%24:0>2d}': np.zeros(6) for n in range(range_start, range_end)}
    data = pd.read_csv(csv_path)

    for i, s in data.iterrows():
        hour = s['Time'][:2]
        if hour in total_frames.keys():
            total_frames[hour] += 1
            total_seconds[hour] += 1 / vid_info['FPS']
            counter_detected[hour] += (np.array(s[1:].tolist()) >= 1)
            counter_on_mouse[hour] += (np.array(s[1:].tolist()) >= 2)

    statistic_detected = {k: counter_detected[k]/(total_frames[k]+1e-6) for k in total_frames.keys()}
    statistic_on_mouse = {k: counter_on_mouse[k]/(total_frames[k]+1e-6) for k in total_frames.keys()}

    # detect
    path = f'{OUTPUT_DIR}/{exp_no}/{exp_no}-count_per_hour_detected.csv'
    with open(path, 'w') as cf:
        write = csv.writer(cf)
        write.writerow(fields_3)
        for k, d in statistic_detected.items():
            d = np.round(d, 3)
            r = [k] + d.tolist() + [total_frames[k], round(total_seconds[k], 2)]
            write.writerow(r)

    detected = np.stack([a for a in statistic_detected.values()]).T
    fig = plt.figure(dpi=150)
    ax = plt.axes()
    for i in range(6):
        plt.plot(detected[i], '-o', alpha=0.6, color=colors[i], label=f'{markers[i]} - {sex[i]}')

    plt.grid(alpha=0.2)
    plt.legend()
    ax.set_xticks([i for i in range(range_end-range_start)])
    ax.set_xticklabels([f'{n%24:0>2d}' for n in range(range_start, range_end)])
    ax.set_xlabel('Hour', fontsize=15)
    fig.savefig(path.replace('.csv', '.png'), pad_inches=1)
    plt.close()

    # on mouse
    path = f'{OUTPUT_DIR}/{exp_no}/{exp_no}-count_per_hour_on_mouse.csv'
    with open(path, 'w') as cf:
        write = csv.writer(cf)
        write.writerow(fields_3)
        for k, d in statistic_on_mouse.items():
            d = np.round(d, 3)
            r = [k] + d.tolist() + [total_frames[k], round(total_seconds[k], 2)]
            write.writerow(r)

    detected = np.stack([a for a in statistic_on_mouse.values()]).T
    fig = plt.figure(dpi=150)
    ax = plt.axes()
    for i in range(6):
        plt.plot(detected[i], '-o', alpha=0.6, color=colors[i], label=f'{markers[i]} - {sex[i]}')

    plt.legend()
    plt.grid(alpha=0.2)
    ax.set_xticks([i for i in range(range_end-range_start)])
    ax.set_xticklabels([f'{n%24:0>2d}' for n in range(range_start, range_end)])
    ax.set_xlabel('Hour', fontsize=15)
    fig.savefig(path.replace('.csv', '.png'), pad_inches=1)
    plt.close()