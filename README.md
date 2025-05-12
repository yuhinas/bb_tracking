# Burying Beetle Tracking and Ranking
This project is designed to track the movement of burying beetles and the position of mouse during behavioural experiments. Based on the tracking results, it calculates the amount of time each beetle spends on the mouse and analyzes beetle tracks to estimate and infer their social rank.

## Required Data
Please download the [required data](https://drive.google.com/drive/folders/1vGJgGVYwupqA06Lj6wFC9Ki-DBtZDRHL?usp=drive_link) first.

Place all model weight files (`.pth` files) in the `tracking/data/` directory. These are required for the project to run.

A sample `.mp4` video is also provided for demonstration purposes. Place it in the `video/n7141/` directory. You may choose to skip this video if you plan to use your own data, but make sure to follow the required folder structure: each experiment should be placed in a separate `nXXXX` folder under `video/`.

Note: Some folders (e.g., `tracking/data/`, `video/n7141`) are included with `.gitkeep` files to ensure structure. Please place your actual data in these directories.

## Features
The tracking pipeline consists of three stages: preprocess, tracking, and postprocess.

### Preprocess

- **video_night_classifier.py**  
  Generates a list of nighttime video files for tracking.  
  The output is a JSON file that lists all videos to be processed.  
  You can set which experiments and time range to include.

### Tracking (Burying Beetle and Mouse Detection)

- **track.py**  
  Detects burying beetles and records mouse positions and body area.  
  Determines whether each beetle is on the mouse in every frame.

- **config.yml**  
  Configuration file used by `track.py`.  
  You must modify `video_dict` and `output_dir` according to your experiment.

- **script_output_tracked_video.ipynb**  
  Generates visualization videos with beetle bounding boxes and additional metrics annotated on frames.

### Postprocess

- **count_on_mouse.py**  
  Calculates how long each beetle stays on the mouse, based on raw tracking data.  
  Also provides hourly summaries.
  You can specify the analysis time range.

- **active_radius_SpecifiedStartEnd.py**  
  Analyzes beetle movement deviation and estimates social ranks.  
  outputs 2D plots of Positional Deviation Index over time for each experiment.  
  You can set which experiments and time range to include.

- **merge_csv.py**  
  Merges all relevant CSV files into a single `merged_data.csv` summary table that can be used for further statistical analysis.

- **tracks_3d_plots_batch_output.py**  
  Generates 3D plots of beetle tracks in batch.

## Requirements


## Installation


## Usage
The full pipeline consists of three stages: Preprocess → Tracking → Postprocess (count_on_mouse, active_radius_analysis → analysis_merge).

All scripts in this project can be executed using the following format:  
```bash
python path/to/script.py
```

The only exception is `tracking/track.py`, which requires a YAML configuration file as an argument:  
```bash
python track.py config.yml
```

(Optional) If you want to directly inspect tracking results from video frames, you can refer to `tracking/script_output_tracked_video.ipynb`. The relevant code cells are provided for reference and should be copied into a `.py` script for execution.

## License
