import pandas as pd
import os
import glob
import numpy as np

RANK_DIR = '../../output/active-radius-analysis'
ON_MOUSE_DIR = '../../output/on-mouse'
MOUSE_AREA_DIR = '../../output/mouse-record'
OUTPUT_DIR = '../../output'

rank_total = pd.read_csv(f'{RANK_DIR}/conclusion.csv')
rank_hour = pd.read_csv(f'{RANK_DIR}/conclusion_hour.csv')

rank_merge = rank_hour.copy()
columns_to_add = rank_total.columns[1:]

for col in columns_to_add:
    rank_merge[f"total_{col}"] = rank_merge["Experiment No."].map(rank_total.set_index("Experiment No.")[col])
    
rank_merge = rank_merge.rename(columns={'Experiment No.': 'exp_no',
                                         'hours': 'hour',
                                         'F. Rank 1': 'F_alpha',
                                         'F. Rank 2': 'F_beta',
                                         'F. Rank 3': 'F_gamma',
                                         'F. Score 1': 'F_alpha_score',
                                         'F. Score 2': 'F_beta_score',
                                         'F. Score 3': 'F_gamma_score',
                                         'M. Rank 1': 'M_alpha',
                                         'M. Rank 2': 'M_beta',
                                         'M. Rank 3': 'M_gamma',
                                         'M. Score 1': 'M_alpha_score',
                                         'M. Score 2': 'M_beta_score',
                                         'M. Score 3': 'M_gamma_score',
                                         'total_F. Rank 1': 'total_F_alpha',
                                         'total_F. Rank 2': 'total_F_beta',
                                         'total_F. Rank 3': 'total_F_gamma',
                                         'total_F. Score 1': 'total_F_alpha_score',
                                         'total_F. Score 2': 'total_F_beta_score',
                                         'total_F. Score 3': 'total_F_gamma_score',
                                         'total_M. Rank 1': 'total_M_alpha',
                                         'total_M. Rank 2': 'total_M_beta',
                                         'total_M. Rank 3': 'total_M_gamma',
                                         'total_M. Score 1': 'total_M_alpha_score',
                                         'total_M. Score 2': 'total_M_beta_score',
                                         'total_M. Score 3': 'total_M_gamma_score'})

# Add new columns for total hour scores
hour_score_columns = [
    'total_F_alpha_hour_score', 'total_F_beta_hour_score', 'total_F_gamma_hour_score',
    'total_M_alpha_hour_score', 'total_M_beta_hour_score', 'total_M_gamma_hour_score'
]
rank_merge = rank_merge.assign(**{col: 0.0 for col in hour_score_columns})

mapping = {
    'F': (['total_F_alpha', 'total_F_beta', 'total_F_gamma'],
          ['F_alpha', 'F_beta', 'F_gamma'],
          ['total_F_alpha_hour_score', 'total_F_beta_hour_score', 'total_F_gamma_hour_score'],
          ['F_alpha_score', 'F_beta_score', 'F_gamma_score']),
    'M': (['total_M_alpha', 'total_M_beta', 'total_M_gamma'],
          ['M_alpha', 'M_beta', 'M_gamma'],
          ['total_M_alpha_hour_score', 'total_M_beta_hour_score', 'total_M_gamma_hour_score'],
          ['M_alpha_score', 'M_beta_score', 'M_gamma_score'])
}

# Map total_* columns to corresponding scores based on individual markers
for idx, row in rank_merge.iterrows():
    for prefix, (total_cols, marker_cols, hour_score_cols, score_cols) in mapping.items():
        for total_col, hour_score_col in zip(total_cols, hour_score_cols):
            individual = row[total_col]
            for marker_col, score_col in zip(marker_cols, score_cols):
                if row[marker_col] == individual:
                    rank_merge.at[idx, hour_score_col] = row[score_col]
                    break

# Drop unused score columns
columns_to_drop = ['F_alpha_score', 'F_beta_score', 'F_gamma_score', 'M_alpha_score', 'M_beta_score', 'M_gamma_score']
rank_merge = rank_merge.drop(columns=columns_to_drop)

# concat on_mouse_data from all exp 
on_mouse_data = []

for exp in os.listdir(ON_MOUSE_DIR):
    csv_file = f'{ON_MOUSE_DIR}/{exp}/{exp}-count_per_hour_on_mouse.csv'
    df = pd.read_csv(csv_file)
    df.insert(0, "Experiment No.", exp)
    on_mouse_data.append(df)
    
all_on_mouse_data = pd.concat(on_mouse_data, ignore_index=True)

# Map the ranks to calculate total seconds for each rank
def calculate_on_mouse_sec(row, rank):
    return round(row[rank] * row['total_sec'], 2)

# Add new columns to the on_mouse DataFrame
for index, rank_row in rank_total.iterrows():
    exp_no = rank_row['Experiment No.']
    
    # Filter rows corresponding to the current experiment
    exp_data = all_on_mouse_data[all_on_mouse_data['Experiment No.'] == exp_no].copy()

    # Map ranks for F and M to the respective columns
    rank_mapping = {
        'F_alpha': rank_row['F. Rank 1'],
        'F_beta': rank_row['F. Rank 2'],
        'F_gamma': rank_row['F. Rank 3'],
        'M_alpha': rank_row['M. Rank 1'],
        'M_beta': rank_row['M. Rank 2'],
        'M_gamma': rank_row['M. Rank 3']
    }

    for col, rank in rank_mapping.items():
        exp_data[f'total_{col}_sec'] = exp_data.apply(lambda row: calculate_on_mouse_sec(row, rank), axis=1)

    # Update the on_mouse DataFrame
    all_on_mouse_data.loc[exp_data.index, exp_data.columns] = exp_data
    
# Merge rank_merge and all_on_mouse_data based on 'exp_no' and 'hour' matching 'Experiment No.' and 'Hour'
rank_merge = rank_merge.merge(
    all_on_mouse_data[['Experiment No.', 'Hour', 'total_sec', 'total_F_alpha_sec', 'total_F_beta_sec', 'total_F_gamma_sec', 'total_M_alpha_sec', 'total_M_beta_sec', 'total_M_gamma_sec']], 
    left_on=['exp_no', 'hour'], 
    right_on=['Experiment No.', 'Hour'], 
    how='left')

rank_merge = rank_merge.drop(columns=['Experiment No.', 'Hour'])
rank_merge = rank_merge.rename(columns={'total_sec': 'video_sec'})

mouse_pixels_data = []

# Loop through each experiment folder
for exp in os.listdir(MOUSE_AREA_DIR):
    exp_folder = os.path.join(MOUSE_AREA_DIR, exp)
    if os.path.isdir(exp_folder):  # Ensure it's a directory
        # Use glob to get all CSV files in the experiment folder
        csv_files = glob.glob(f'{exp_folder}/*.csv')
        
        for csv_file in csv_files:
            # Read each CSV file
            df = pd.read_csv(csv_file)
            # Add a column for the experiment number
            df.insert(0, "exp_no", exp)
            # Append the DataFrame to the list
            mouse_pixels_data.append(df)

# Concatenate all the data frames into one
all_mouse_pixels_data = pd.concat(mouse_pixels_data, ignore_index=True)

# Convert 'time' column to datetime with format HH:MM:SS
all_mouse_pixels_data['time'] = pd.to_datetime(all_mouse_pixels_data['time'], format='%H:%M:%S', errors='coerce')

# Check if there are any rows where the conversion failed
failed_rows = all_mouse_pixels_data[all_mouse_pixels_data['time'].isna()]
if not failed_rows.empty:
    print('The following rows failed time format conversion:')
    print(failed_rows)
else:
    print('No conversion failures')

# Extract the hour part from the 'time' column
all_mouse_pixels_data['hour'] = all_mouse_pixels_data['time'].dt.hour

# Find the earliest time for each combination of 'exp_no' and 'hour'
earliest_pixels = all_mouse_pixels_data.loc[all_mouse_pixels_data.groupby(['exp_no', 'hour'])['time'].idxmin()]

# Merge the earliest 'area_pixels' values into rank_merge based on 'exp_no' and 'hour'
rank_merge = rank_merge.merge(
    earliest_pixels[['exp_no', 'hour', 'area_pixels']], 
    on=['exp_no', 'hour'], 
    how='left'
)

# Rename the merged 'area_pixels' column to 'mouse_pixels'
rank_merge = rank_merge.rename(columns={'area_pixels': 'mouse_pixels'})

# Reorder columns
output_csv_columns = ['exp_no', 'hour',
                      'F_alpha', 'F_beta', 'F_gamma', 'M_alpha', 'M_beta', 'M_gamma',
                      'total_F_alpha', 'total_F_beta', 'total_F_gamma',
                      'total_F_alpha_score', 'total_F_beta_score', 'total_F_gamma_score',
                      'total_F_alpha_hour_score', 'total_F_beta_hour_score', 'total_F_gamma_hour_score',
                      'total_M_alpha', 'total_M_beta', 'total_M_gamma',
                      'total_M_alpha_score', 'total_M_beta_score', 'total_M_gamma_score',
                      'total_M_alpha_hour_score', 'total_M_beta_hour_score', 'total_M_gamma_hour_score',
                      'total_F_alpha_sec', 'total_F_beta_sec', 'total_F_gamma_sec',
                      'total_M_alpha_sec', 'total_M_beta_sec', 'total_M_gamma_sec',
                      'video_sec', 'mouse_pixels']

rank_merge = rank_merge[output_csv_columns]

# Format float values to avoid scientific notation and remove trailing zeros
rank_merge = rank_merge.applymap(lambda x: ("{:.6f}".format(x).rstrip('0').rstrip('.') if isinstance(x, (float, np.float64)) else x))

# Save the DataFrame to CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = f'{OUTPUT_DIR}/merged_data.csv'
rank_merge.to_csv(csv_path, index=False)

print(f'Program execution completed. The output file has been saved to: {csv_path}')