from glob import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import json
import re
import os

# path
RAW_TRACK_DIR = '../../output/raw-tracks'
OUTPUT_DIR = '../../output/tracks_3d_plots'

# fixed config
markers = ['O', 'X', 'nn', 'H', 'ss', 'xx']
sex = ['F', 'F', 'F', 'M', 'M', 'M']
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# function
def get_time(path):
    """
    For file name in format '[Chanel] date %H.%M.%S'.
    """
    t_regex = re.compile(r'\d\d\.\d\d\.\d\d')
    t = t_regex.search(path).group()
    t = list(map(int, t.split('.')))
    return t[0]*3600 + t[1]*60 + t[2]
    
def box2xy(boxes:np.array):
    xy = np.empty((len(boxes), 2))
    xy[:,0] = np.mean(boxes[:,[0, 2]], axis=1)
    xy[:,1] = np.mean(boxes[:,[1, 3]], axis=1)
    return xy

def second2time(f_no:np.array, fps, init_time):
    second = f_no / fps
    second = second + init_time
    second = second % 86400  # align 24 hours
    h = (second // 3600).astype(int)
    m = (second % 3600 // 60).astype(int)
    s = (second % 60).astype(int)
    t = [f'{i:0>2d}:{j:0>2d}:{k:0>2d}' for i, j, k in zip(h, m, s)]
    return t

# plot config
EXP_LIST = 'all'  # List of experiment no. or 'all' for all of experiment in RAW_TRACK_DIR
INTERVAL = 300  # second
LINK_THRESHOLD = 30  # second


if EXP_LIST == 'all':
    EXP_LIST = list(map(os.path.basename,glob(f'{RAW_TRACK_DIR}/*')))

for exp_no in tqdm(EXP_LIST):

    # create output dir
    output_dir = f'{OUTPUT_DIR}/{exp_no}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    json_list = sorted(glob(f'{RAW_TRACK_DIR}/{exp_no}/*json'))
    for json_path in json_list:
        with open(json_path) as jf:
            data = json.load(jf)

        ### temp
        data['FPS'] = 15.
        ###

        init_time = get_time(json_path)
        tracks = {m: np.empty((2, data['Video']['Length'])) for m in markers}

        for f_no, frame_data in enumerate(data['Tracks']):
            if frame_data['Boxes'] != []:
                boxes = np.array(frame_data['Boxes'])
                xy = box2xy(boxes)

            for m in markers:
                if m in frame_data['Labels']:
                    ndx = frame_data['Labels'].index(m)
                    tracks[m][:, f_no] = xy[ndx]
                else:
                    tracks[m][:, f_no] = np.nan
                    
        # create figure
        n_plot = np.ceil(data['Video']['Length']/(data['FPS']*INTERVAL)).astype(int)
        fig = plt.figure(figsize=(12, n_plot*15), dpi=200)

        for j in range(n_plot):

            ax = fig.add_subplot(n_plot, 1, n_plot-j, projection='3d')

            s_begin = 300*j
            f_begin = int(s_begin * data['FPS'])
            f_end = int((s_begin+INTERVAL) * data['FPS'])
            ts = np.arange(f_begin, min(f_end, data['Video']['Length']))
            
            # plot
            for i, m in enumerate(markers):

                xs = np.array(tracks[m][0][f_begin:f_end])
                ys = np.array(tracks[m][1][f_begin:f_end])
                mask = ~np.isnan(xs)

                ax.scatter(xs[mask], ys[mask], ts[mask],s=7 , color=colors[i], alpha=0.6, label=f'{m} - {sex[i]}')

            formatter = FuncFormatter(lambda x, pos: f'{x:,.0f}')

            ax.set_xlim(0, data['Video']['Width'])
            ax.set_ylim(0, data['Video']['Height'])
            ax.set_zlim(f_begin, f_end)
            ax.set_box_aspect((4, 3, 10))

            # z ticks
            ticks = np.linspace(f_begin, f_end, 11)
            hms = second2time(ticks, data['FPS'], init_time)
            ax.zaxis.set_ticks(ticks)
            ax.zaxis.set_ticklabels(hms, fontsize=12, ha='left')
            
            # xy tircks
            ax.xaxis.set_ticks([tick for tick in range(0, 1920, 250)])
            ax.yaxis.set_ticks([tick for tick in range(0, 1080, 250)])
            ax.xaxis.set_ticklabels([str(tick) for tick in range(0, 1920, 250)], fontsize=12, ha='right')
            ax.yaxis.set_ticklabels([str(tick) for tick in range(0, 1080, 250)], fontsize=12, ha='left')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis='x', pad=0, labelrotation=45)
            ax.tick_params(axis='y', pad=0, labelrotation=-20)
            
            ax.set_xlabel('X (pixel)', labelpad=18, fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (pixel)', labelpad=15, fontsize=12, fontweight='bold')
            ax.set_zlabel('T (h:m:s)', labelpad=80, fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', markerscale=1.5, fontsize=12)

            if n_plot-j == 1:
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                fig_title = base_name.replace(".", " ")
                ax.set_title(fig_title, pad=25)
                
#         plt.show()
#         raise Exception

        fig.savefig(f'{output_dir}/{fig_title}.jpg', bbox_inches='tight', pad_inches=1, dpi=200)
        plt.close()