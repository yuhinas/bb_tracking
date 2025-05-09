from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import json
import os
import re
from datetime import datetime, timedelta

### config
RAW_TRACK_DIR = '../../output/raw-tracks'
OUTPUT_DIR = '../../output/active-radius-analysis'
JSON_FILTER = ['*']  
              # List of exp ['n10239', 'n10240'], or for all exp: ['*']
PERCENTILE = 95

### Analysis time window
START_TIME = "20:00:00"
END_TIME = "05:00:00"

### fixed config
markers = ['O', 'X', 'nn', 'H', 'ss', 'xx']
markers_with_loc = ['O x-axis', 'O y-axis', 'O dis_index',
                    'X x-axis', 'X y-axis', 'X dis_index',
                    'nn x-axis', 'nn y-axis', 'nn dis_index',
                    'H x-axis', 'H y-axis', 'H dis_index',
                    'ss x-axis', 'ss y-axis', 'ss dis_index',
                    'xx x-axis', 'xx y-axis', 'xx dis_index']
sex = ['F', 'F', 'F', 'M', 'M', 'M']
colors = ['r', 'g', 'b', 'c', 'm', 'y']
fields = ['F. Rank 1', 'F. Rank 2', 'F. Rank 3',
          'F. Score 1', 'F. Score 2', 'F. Score 3',
          'M. Rank 1', 'M. Rank 2', 'M. Rank 3',
          'M. Score 1', 'M. Score 2', 'M. Score 3']

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
    """
    Convert bounding boxes to center coordinates.
    """
    xy = np.empty((len(boxes), 2))
    xy[:, 0] = np.mean(boxes[:,[0, 2]], axis=1)
    xy[:, 1] = np.mean(boxes[:,[1, 3]], axis=1)
    return xy

def second2time(f_no:np.array, fps, init_time):
    """
    Convert frame number to time format (hh:mm:ss).
    """
    second = f_no / fps
    second = second + init_time
    second = second % 86400  # align 24 hours
    h = (second // 3600).astype(int)
    m = (second % 3600 // 60).astype(int)
    s = (second % 60).astype(int)
    t = [f'{i:0>2d}:{j:0>2d}:{k:0>2d}' for i, j, k in zip(h, m, s)]
    return t

def filter_time_window(time_list, start_time_filter, end_time_filter):
    """
    Filter indices of data within the specified time window.
    Handles time ranges that cross midnight.
    """
    indices = []
    for i, t in enumerate(time_list):
        t_obj = datetime.strptime(t, "%H:%M:%S").time()
        
        # Check time range
        if start_time_filter <= end_time_filter:  # Same-day time range
            if start_time_filter <= t_obj < end_time_filter:
                indices.append(i)
        else:  # Time range that crosses midnight
            if t_obj >= start_time_filter or t_obj < end_time_filter:
                indices.append(i)
    return indices

exp_list = []
for exp in JSON_FILTER:
    exp_list.extend(glob(f'{RAW_TRACK_DIR}/{exp}'))
exp_list = sorted(set(exp_list))

conclusion = []
conclusion_hour = []
except_list = []

for tar_path in exp_list:
    exp_no = os.path.basename(tar_path)
    out_dir = f'{OUTPUT_DIR}/{exp_no}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    json_list = sorted(glob(f'{tar_path}/*.json'))
    fps = 0
    date = json_list[0].split('/')[-1].split()[1]
    time = json_list[0].split('/')[-1].split()[2][:8]
    dt_base = datetime.strptime(f'{date} {time}', '%Y-%m-%d %H.%M.%S')

    # Convert START and END times for filtering
    actual_start_time = datetime.strptime(json_list[0].split("/")[-1].split('.')[0][-2:] + ":" +
                                          json_list[0].split("/")[-1].split('.')[1][:2] + ":" +
                                          json_list[0].split("/")[-1].split('.')[2][:2], "%H:%M:%S").time()
    start_time_filter = max(datetime.strptime(START_TIME, "%H:%M:%S").time(), actual_start_time)
    end_time_filter = datetime.strptime(END_TIME, "%H:%M:%S").time()

    # ---
    # Get data
    # ---
    time_line, xy_tracks, xy_with_loc = [], [], []

    for path in tqdm(json_list, desc=exp_no):
        date_current = path.split('/')[-1].split()[1]
        time_current = path.split('/')[-1].split()[2][:8]
        dt_current = datetime.strptime(f'{date_current} {time_current}', '%Y-%m-%d %H.%M.%S')
        if (dt_current - dt_base).total_seconds() >= 60*60*18:
            continue
        with open(path) as jf:
            data = json.load(jf)

        vid_info = data['Video']
        if vid_info['FPS'] != 0 and fps == 0:
             fps =  vid_info['FPS']
        objs = data['Tracks']
        init_time = get_time(path)
        t = np.array([i for i in range(len(objs))])
        t = second2time(t, vid_info['FPS'], init_time)
        time_line += t

        for info in objs:
            xy = np.empty((len(markers), 2))
            xy[:] = np.nan

            if info['Boxes']:
                indices = []
                for m in info['Labels']:
                    indices.append(markers.index(m))

                boxes = np.array(info['Boxes'])
                xy[indices] = [[f'{box2xy(boxes)[0][0]:.6f}', f'{box2xy(boxes)[0][1]:.6f}']]
            xy_tracks.append(xy)

            # container with loc
            xy_loc = np.empty(len(markers_with_loc))
            xy_loc[:] = np.nan
            if info['Boxes']:
                for m in info['Labels']:
                    xy_loc[markers.index(m)*3] = f"{np.mean([info['Boxes'][0][0],info['Boxes'][0][2]]):.6f}"
                    xy_loc[markers.index(m)*3+1] = f"{np.mean([info['Boxes'][0][1],info['Boxes'][0][3]]):.6f}"
            xy_with_loc.append(xy_loc)

    # Filter data based on time window
    filter_indices = filter_time_window(time_line, start_time_filter, end_time_filter)
    xy_tracks = np.array(xy_tracks)[filter_indices]
    xy_with_loc = np.array(xy_with_loc)[filter_indices]
    time_line = [time_line[i] for i in filter_indices]

    if len(xy_tracks) == 0:
        continue

    # ---
    # Distance Calculations
    # ---
    try:
        xy_tracks = np.array(xy_tracks)
        self_centers = np.nanmean(xy_tracks, axis=0)
        self_deviations = xy_tracks - self_centers
        self_distances = np.sqrt(np.sum(self_deviations**2, axis=-1))

        f_centers = np.nanmean(xy_tracks[:, :3].reshape(-1, 2), axis=0)
        m_centers = np.nanmean(xy_tracks[:, 3:].reshape(-1, 2), axis=0)
        f_deviations = self_centers[:3] - f_centers
        m_deviations = self_centers[3:] - m_centers
        sex_deviations = np.concatenate([f_deviations, m_deviations], axis=0)
        sex_distances = np.sqrt(np.sum(sex_deviations**2, axis=-1))

        distances = self_distances + sex_distances
    except:
        print(f"Error: {tar_path}")
        continue

    # distances = np.round(distances.astype(float), 6)
    for idx, i in enumerate(distances):
        for jdx, jdata in enumerate(i):
            distances[idx][jdx] = f'{jdata:.6f}'

    # # ---
    # # Output dis_index CSV
    # # ---
    # data = pd.DataFrame(distances, columns=markers)
    # data.insert(0, 'Time', time_line)
    # data.to_csv(f'{out_dir}/dis_index.csv')

    # # ---
    # # Output dis_index CSV with beetle location
    # # ---
    # for idx, i in enumerate(distances):
    #     for jdx, jdata in enumerate(i):
    #         xy_with_loc[idx][jdx*3+2] = jdata 
    # data = pd.DataFrame(xy_with_loc, columns=markers_with_loc)
    # data.insert(0, 'Time', time_line)
    # data.to_csv(f'{out_dir}/dis_index_with_loc.csv')

    # ---
    # Summary
    # ---
    total_avail_data = (~np.isnan(np.transpose(distances))).sum(1)

    f_radius = np.nanpercentile(distances[:, :3], PERCENTILE)
    m_radius = np.nanpercentile(distances[:, 3:], PERCENTILE)

    f_occurrences = distances[:, :3] < f_radius
    m_occurrences = distances[:, 3:] < m_radius

    f_scores = np.sum(f_occurrences, axis=0) / (len(distances)+1e-6)
    m_scores = np.sum(m_occurrences, axis=0) / (len(distances)+1e-6)

    f_total = np.sum(f_occurrences, axis=0)
    m_total = np.sum(m_occurrences, axis=0)

    f_ranks = sorted(
        [[m, f'{s:.6f}', t, tt] for m, s, t, tt in zip(markers[:3], f_scores, f_total, total_avail_data[:3])],
        key=lambda x: x[1], reverse=True
    )
    m_ranks = sorted(
        [[m, f'{s:.6f}', t, tt] for m, s, t, tt in zip(markers[3:], m_scores, m_total, total_avail_data[3:])],
        key=lambda x: x[1], reverse=True
    )

    f_data = [f[0] for f in f_ranks] + [f[1] for f in f_ranks]
    m_data = [m[0] for m in m_ranks] + [m[1] for m in m_ranks]
    result = f_data + m_data

    conclusion.append([exp_no] + result)

    # ---
    # Hourly Summary
    # ---
    # Create a mapping of time_line to corresponding hours
    hour_bins = [datetime.strptime(t, "%H:%M:%S").hour for t in time_line]
    start_hour = hour_bins[0]
    end_hour = hour_bins[-1]
    
    # Handle hours across midnight
    if start_hour > end_hour:
        hours_range = list(range(start_hour, 24)) + list(range(0, end_hour + 1))
    else:
        hours_range = list(range(start_hour, end_hour + 1))
    
    for hour in hours_range:
        # Filter data belonging to the current hour
        hour_indices = [i for i, h in enumerate(hour_bins) if h == hour]
        # print(f"Hour {hour}: Total entries = {len(hour_indices)}")  # Debug message
        
        if not hour_indices:
            continue
        
        distances_hour = distances[hour_indices]
        f_occurrences_hour = distances_hour[:, :3] < f_radius
        m_occurrences_hour = distances_hour[:, 3:] < m_radius
        
        # Calculate scores and total occurrences for current hour
        f_scores_hour = np.sum(f_occurrences_hour, axis=0) / (len(distances_hour) + 1e-6)
        m_scores_hour = np.sum(m_occurrences_hour, axis=0) / (len(distances_hour) + 1e-6)
        f_total_hour = np.sum(f_occurrences_hour, axis=0)
        m_total_hour = np.sum(m_occurrences_hour, axis=0)
        
        # Sort ranks for female and male markers
        f_ranks_hour = sorted(
            [[m, f'{s:.6f}', t] for m, s, t in zip(markers[:3], f_scores_hour, f_total_hour)],
            key=lambda x: x[1], reverse=True
        )
        m_ranks_hour = sorted(
            [[m, f'{s:.6f}', t] for m, s, t in zip(markers[3:], m_scores_hour, m_total_hour)],
            key=lambda x: x[1], reverse=True
        )
        
        # Store data for the current hour
        f_data_hour = [f[0] for f in f_ranks_hour] + [f[1] for f in f_ranks_hour]
        m_data_hour = [m[0] for m in m_ranks_hour] + [m[1] for m in m_ranks_hour]
        conclusion_hour.append([exp_no] + [hour] + f_data_hour + m_data_hour)

    # ---
    # Plot
    # ---    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=150)
    
    # Convert START_TIME and END_TIME to datetime objects
    start_time_dt = datetime.strptime(START_TIME, "%H:%M:%S")
    end_time_dt = datetime.strptime(END_TIME, "%H:%M:%S")
    if end_time_dt <= start_time_dt:  # Handle crossing midnight
        end_time_dt += timedelta(days=1)
    
    # Generate full hour intervals from START_TIME to END_TIME
    full_hours = []
    current_time = start_time_dt
    while current_time <= end_time_dt:
        full_hours.append(current_time)
        current_time += timedelta(hours=1)
    
    # Convert time_line to datetime and calculate index positions
    time_line_dt = [datetime.strptime(t, "%H:%M:%S") for t in time_line]
    xticks = []
    xticklabels = []
    
    # Mapping for start time
    start_index = 0
    xticks.append(start_index)
    xticklabels.append(time_line_dt[0].strftime("%H:%M"))
    
    # Map full_hours to the closest available positions in time_line
    for hour in full_hours[1:-1]:
        closest_index = None
        for i, t in enumerate(time_line_dt):
            if t.hour == hour.hour and t.minute == hour.minute:  # Exact match
                closest_index = i
                break
            elif t > hour and closest_index is None:  # Use the first future match
                closest_index = i
        if closest_index is None:
            closest_index = len(time_line_dt) - 1  # Default to last index if no match
        xticks.append(closest_index)
        xticklabels.append(hour.strftime("%H:%M"))
    
    # Mapping for end time (rounded up to the next hour if minute == 59)
    if time_line_dt[-1].minute == 59 and time_line_dt[-1].second == 59:
        time_line_dt[-1] = (time_line_dt[-1]+ timedelta(minutes=1)).replace(minute=0, second=0)
    end_index = len(time_line_dt) - 1
    xticks.append(end_index)
    xticklabels.append(time_line_dt[-1].strftime("%H:%M"))
    
    # Set Y-axis number format: comma as thousands separator
    formatter = FuncFormatter(lambda x, pos: f'{x:,.0f}')
    
    # Sort markers, corresponding colors, and data rows based on rank results
    ranked_markers = [f[0] for f in f_ranks] + [m[0] for m in m_ranks]
    ranked_colors = [colors[markers.index(m)] for m in ranked_markers]
    ranked_data = [distances[:, markers.index(m)] for m in ranked_markers]

    # Female plot
    axes[0].set_title('Female')
    axes[0].set_ylabel('Positional Deviation Index')
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].set_prop_cycle(color=ranked_colors[:3])
    plots = [axes[0].plot(d, alpha=0.6)[0] for d in ranked_data[:3]]
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels)
    axes[0].legend(plots, ranked_markers[:3], loc='upper right')
    axes[0].grid(alpha=0.4)
    axes[0].axhline(f_radius, color='k', linestyle='--')
    axes[0].text(xticks[0], 1.05*f_radius, f'Activity Radius')
    axes[0].text(xticks[0], 0.85*f_radius, f'({PERCENTILE}th Percentile)')

    # Male plot
    axes[1].set_title('Male')
    axes[1].set_xlabel('Time (h:m)')
    axes[1].set_ylabel('Positional Deviation Index')
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].set_prop_cycle(color=ranked_colors[3:])
    plots = [axes[1].plot(d, alpha=0.6)[0] for d in ranked_data[3:]]
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticklabels)
    axes[1].legend(plots, ranked_markers[3:], loc='upper right')
    axes[1].grid(alpha=0.4)
    axes[1].axhline(m_radius, color='k', linestyle='--')
    axes[1].text(xticks[0], 1.05*m_radius, f'Activity Radius')
    axes[1].text(xticks[0], 0.85*m_radius, f'({PERCENTILE}th Percentile)')

    fig.tight_layout()
    fig.savefig(f'{out_dir}/plot.png')
    plt.close()

# ---
# Output conclusion CSV
# ---
fields2 = ['Experiment No.'] + fields
df = pd.DataFrame(conclusion, columns=fields2)
df.to_csv(f'{OUTPUT_DIR}/conclusion.csv', index=False)
print('Finish the conclusion.')
if except_list:
    print("Below list has exception:")
    for i in except_list:
        print(i)
# ---
# Output hourly conclusion CSV
# ---
fields_hour = ['Experiment No.', 'hours'] + fields
df = pd.DataFrame(conclusion_hour, columns=fields_hour)
df.to_csv(f'{OUTPUT_DIR}/conclusion_hour.csv', index=False)
print('Finish the hour conclusion.')