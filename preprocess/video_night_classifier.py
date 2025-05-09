from glob import glob
from datetime import datetime, timedelta
import json
import re

# Configuration
TARGET_DIR = '../video/'  # Target directory, please use ABSOLUTE paths
OUTPUT_JSON = '../output/video_list/tracking_list.json'  # Output filename
target_dir = []
             # Target directories without "n", if just need to handle certain directories
             # Example: only process folders n7141 and n7142 → set as [7141, 7142]
             # Leave as [] to process all available directories

# Directory range without "n"
dir_range_min = 1
dir_range_max = 99999000

start_hr = 18
end_hr = 5
# ------ #
# Long photoperiod = 19:00:00-05:00:00 (strat_hr=18, end_hr=5)
# Short photoperiod = 17:00:00-07:00:00 (strat_hr=16, end_hr=7)
# Start one hour earlier to ensure the target analysis period is fully covered.
# ------ #

start_day = 0
end_day = 1
# ------ #
# night1: start_day=0, end_day=1
# night2: start_day=1, end_day=2
# night1 + night2: start_day=0, end_day=2
# ------ #

def catch_base_dir(dir_name):
    """
    Extract the numeric base part of the directory name (e.g., '7420' from '7420_a' or '7420-1').
    """
    match = re.search(r'\d+', dir_name)  # Match numeric part in directory name
    return match.group() if match else None

def catch_base_dir_HSP(dir_name):
    """
    Extract numeric ID from the third segment (e.g., 'n8696' → '8696') in folder name like '17_MR_n8696_no2nd'.
    """
    parts = dir_name.split('_')
    if len(parts) >= 3 and parts[2].startswith('n') and parts[2][1:].isdigit():
        return parts[2][1:]
    return None

def catch_time_in_file(s, p):
    """
    Extract date and time from filename using regex
    """
    result = p.search(s)
    return result.group(1), result.group(2)

video_dict = {}
pattern = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2}\.\d{2}\.\d{2})")

for i in glob(TARGET_DIR + '*'):
    dir_label = i.split(TARGET_DIR)[1]  # Extract the full directory name
    base_dir = catch_base_dir(dir_label)  # Extract numeric base part
#     base_dir = catch_base_dir_HSP(dir_label)  # Extract numeric base part
    if base_dir is None:  # Skip if no valid base number is found
        print(f"Skipping directory: {i} (no valid base number found)")
        continue

    if int(base_dir) <= dir_range_max and int(base_dir) >= dir_range_min:
        # If target directory is set and current directory is not in the target list, skip it.
        if len(target_dir) != 0 and base_dir not in map(str, target_dir):  # Ensure compatibility with target_dir
            continue

        all_video = glob(i + '/*.mp4') + glob(i + '/*.avi')
        all_video = sorted(all_video)
        if not all_video:
            continue

        # Build video_time_map for all videos
        video_time_map = {}
        first_night_video_time = None
        for v in all_video:
            if v[-4:] not in ['.mp4', '.avi']:
                continue
            try:
                date, time = catch_time_in_file(v, pattern)
                video_time = datetime.strptime(f'{date} {time}', '%Y-%m-%d %H.%M.%S')
                video_time_map[v] = video_time
                # Find the earliest night-time video to define base night
                if (video_time.hour >= start_hr) or (video_time.hour < end_hr):
                    if (first_night_video_time is None) or (video_time < first_night_video_time):
                        first_night_video_time = video_time
            except:
                continue

        if first_night_video_time is None:
            print(f"No night-time videos found in: {dir_label}")
            continue

        # Create night time ranges for multiple nights
        night_start = first_night_video_time.replace(hour=start_hr, minute=0, second=0)
        night_end = first_night_video_time.replace(hour=end_hr, minute=0, second=0)
        if night_end <= night_start:
            night_end += timedelta(days=1)

        night_ranges = []
        for night_offset in range(start_day, end_day):
            this_start = night_start + timedelta(days=night_offset)
            this_end = night_end + timedelta(days=night_offset)
            night_ranges.append((this_start.timestamp(), this_end.timestamp()))

        dict_key = dir_label
#         dict_key = f"n{base_dir}"
        video_dict[dict_key] = []

        for v, video_time in video_time_map.items():
            video_timestamp = video_time.timestamp()
            if not any(ts_min <= video_timestamp <= ts_max for ts_min, ts_max in night_ranges):
                continue
            video_dict[dict_key].append(v)

# Save the results to a JSON file
with open(OUTPUT_JSON, 'w') as f:
    json.dump(video_dict, f, indent=2)

print(f"Video selection completed. Results saved in {OUTPUT_JSON}")