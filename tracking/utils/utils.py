import csv
import sys
import os


def create_dir(dir_path, check_overwrite=True):
    if os.path.exists(dir_path):
        if check_overwrite:
            keep_going = input(f'"{dir_path}" exists. Do you want to overwrite it (y/[n])? ')
            if keep_going.casefold() != 'y':
                sys.exit()
    else:
        os.makedirs(dir_path)

def keep_training_log(log_dir, fields:list, data:zip):
    fn = f'{log_dir}/training_log.csv'
    with open(fn, 'w') as cf:
        writer = csv.writer(cf) 
        writer.writerow(fields)
        for d in data:
            writer.writerow(d)
            
def keep_summary_to_txt(log_dir, summary:dict):
    fn = f'{log_dir}/summary.txt'
    with open(fn, 'w') as tf:
        for key, value in summary.items():
            tf.write(f'{key}: {value}\n')
            
def read_txt_to_list(txt_file):
    with open(txt_file) as tf:
        data = tf.readlines()
        data = [d.strip('\n') for d in data]
    return data

def crop_images(pil_image, boxes:list):
    """
    Args:
        pil_image: Input PIL image.
        boxes: List of boxes formated in [left, top, right, bottom]
    """
    cropped_imgs = []
    for b in boxes:
        cropped_imgs.append(pil_image.crop(b))
    return cropped_imgs