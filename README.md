# Burying Beetle Tracking and Ranking
This project is designed to track the movement of burying beetles and the position of mouse during behavioural experiments. Based on the tracking results, it calculates the amount of time each beetle spends on the mouse and analyzes beetle tracks to estimate and infer their social rank.

The tracking pipeline is composed of three modules: U-Net Segmentor for mouse segmentation, YOLOv4 Detector for bounding box extraction, and ResNet-based Classifier for assigning labels to detected objects.

## Required Data
Please download the [required data](https://drive.google.com/drive/folders/1vGJgGVYwupqA06Lj6wFC9Ki-DBtZDRHL?usp=drive_link) first.

Place all model weight files (`.pth` files) in the `tracking/data/` directory. These are required for the project to run.

A sample `.mp4` video is also provided for demonstration purposes. Place it in the `video/n7141/` directory (The folder `video/n7141` is included with `.gitkeep` file to ensure structure). You may choose to skip this video if you plan to use your own data, but make sure to follow the required folder structure: each experiment should be placed in a separate `nXXXX` folder under `video/`.

You can use the pre-trained models to perform tracking directly. However, if you wish to train the model yourself, download `training_dataset.zip` from the required data, unzip it, and replace the existing `tracking/training_dataset/` directory. The current folder only contains a sample to illustrate the structure.

The LabelMe JSON files are not used for training. They are intermediate annotation results created during dataset preparation and are retained here to help illustrate the labeling process.

## Features
The tracking pipeline consists of three stages: preprocess, tracking, and postprocess.

### Preprocess

- **`video_night_classifier.py`**  
  Generates a list of nighttime video files for tracking.  
  The output is a JSON file that lists all videos to be processed.  
  You can set which experiments and time range to include.

### Tracking (Burying Beetle and Mouse Detection)  
You may skip `train.py` and `assess.py` by directly using the pre-trained models with track.py for tracking.

- **`train.py`**  
  Unified training script that supports U-Net for segmentation, YOLOv4 for object detection, and ResNet for classification.
  
- **`assess.py`**  
  Unified evaluation script for three model types:  
  - Segmenter  
    Computes mean Intersection over Union (mIoU) between predicted and ground truth masks. Outputs include a histogram of mIoU scores and a CSV summary.
  - Detector  
    Assesses object detection performance using precision-recall analysis. Matches predicted boxes with ground truth via IoU thresholding and outputs a Precision–Recall curve.  
  - Classifier  
    Evaluates classification accuracy and generates a normalized confusion matrix with per-class performance. Visualization is provided as a heatmap-style matrix.
   
- **`track.py`**  
  Detects burying beetles and records mouse positions and body area.  
  Determines whether each beetle is on the mouse in every frame.

- **`config.yml`**  
  Configuration file used by `train.py`, `assess.py`, `track.py`.  
  You must modify `video_dict` and `output_dir` according to your experiment.

- **`script_output_tracked_video.ipynb`**  
  Generates visualization videos with beetle bounding boxes and additional metrics annotated on frames.

### Postprocess
- **`count_on_mouse.py`**  
  Calculates how long each beetle stays on the mouse, based on raw tracking data.  
  Also provides hourly summaries.  
  You can specify the analysis time range.

- **`active_radius_SpecifiedStartEnd.py`**  
  Analyzes beetle movement deviation and estimates social ranks.  
  Outputs 2D plots of Positional Deviation Index over time for each experiment.  
  You can set which experiments and time range to include.

- **`merge_csv.py`**  
  Merges all relevant CSV files into a single `merged_data.csv` summary table that can be used for further statistical analysis.

- **`tracks_3d_plots_batch_output.py`**  
  Generates 3D plots of beetle tracks in batch.

## Requirements


## Installation


## Usage
The full pipeline consists of three stages: Preprocess → Tracking → Postprocess (count_on_mouse, active_radius_analysis → analysis_merge).

- **`tracking/track.py`**  
  Requires a YAML configuration file.  
  ```bash
  python track.py config.yml
  ```

- **`train.py`**  
  Requires model type and a YAML configuration file. An optional comment can be appended to output filenames.  
  ```bash
  python train.py [segmenter | detector | classifier] config.yml [optional_comment]
  ```

- **`assess.py`**  
  Similar to `train.py`, used for model evaluation.  
  ```bash
  python assess.py [segmenter | detector | classifier] config.yml [optional_comment]
  ```

- **All other scripts**  
  All other scripts in this project can be executed using the following format:  
  ```bash
  python path/to/script.py
  ```

- If you want to directly inspect tracking results from video frames, you can refer to `tracking/script_output_tracked_video.ipynb`. The relevant code cells are provided for reference and should be copied into a `.py` script for execution.

## License
