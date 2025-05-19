from torch.utils.data import DataLoader
from glob import glob
import random
import yaml
import sys
import os

from utils import utils, dataset, trainer

# ---
# python train.py [model_type] [config.yaml] [optional_comment]
#
# [model_type]: segmenter, classifier, detector
# [optional_comment]: Optional comment text that will be appended to the output file names for distinction.
# ---

if len(sys.argv) == 1:
    print('Please assign the model to train.')
    sys.exit()
elif len(sys.argv) == 2:
    print('Please give the config path for the model.')
    sys.exit()

with open(sys.argv[2]) as yf:
    configs = yaml.safe_load(yf)
    
if len(sys.argv) >= 4:
    comment = '-' + '-'.join(sys.argv[3:])
else:
    comment = ''
    
# set device
if configs['device'] == 'cpu':
    device = 'cpu'
else:
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs['device'])
    
models = ['segmenter', 'classifier', 'detector']
assert sys.argv[1] in models, f'Unavailable model. Please select from {models}.'
    
    
if sys.argv[1] == 'segmenter':
    from nets import segmenter
    config = configs['Segmenter']

    data_list = utils.read_txt_to_list(config['training_data_list'])
    random.shuffle(data_list)
    cut = int(config['validation_ratio']*len(data_list))
    train_list = data_list[cut:]
    valid_list = data_list[:cut]

    train_data = dataset.MouseDataset(train_list, config, is_train=True)
    train_loader = DataLoader(
        train_data, 
        batch_size = config['training_batch_size'],
        shuffle =True,
        num_workers = config['num_workers']
    )
    valid_data = dataset.MouseDataset(valid_list, config, is_train=False)
    valid_loader = DataLoader(
        valid_data,
        batch_size = config['training_batch_size'],
        shuffle= False,
        num_workers = config['num_workers']
    )

    test_list = random.choices(valid_list, k=config['n_quick_results'])
    
    if config['model'] == 'default':
        model = segmenter.UNet()
    elif config['model'] == 'official':
        model = segmenter.official_model()

    unet = trainer.UNetTrainer(
        model = segmenter.UNet(),
        train_loader = train_loader,
        valid_loader = valid_loader,
        config = config,
        quick_result = test_list,
        device = device,
        comment = f'unet{comment}'
    )

    unet.fit()
    
    
elif sys.argv[1] == 'classifier':
    from nets import classifier
    config = configs['Classifier']
    
    folder_list = utils.read_txt_to_list(config['training_data_folders'])
    data_list = []
    for d in folder_list:
        data_list += glob(f'{d}/*/*[.jpg, .png]')
    
    random.shuffle(data_list)
    cut = int(config['validation_ratio']*len(data_list))
    train_list = data_list[cut:]
    valid_list = data_list[:cut]    

    # data
    train_data = dataset.BeetleDataset(
        data_list = train_list,
        num_classes = len(configs['classes']),
        config = config,
        is_train = True
    )

    train_loader = DataLoader(
        train_data,
        batch_size = config['training_batch_size'],
        shuffle = True,
        num_workers = config['num_workers']
    )

    valid_data = dataset.BeetleDataset(
        data_list = valid_list,
        num_classes = len(configs['classes']),
        config = config,
        is_train = False
    )
    
    valid_loader = DataLoader(
        valid_data,
        batch_size = config['training_batch_size'],
        shuffle = False,
        num_workers = config['num_workers']
    )
    
    # model
    n_classes = len(configs['classes'])
    if config['model'] == 'default':
        model = classifier.ResNet(n_classes)
    else:
        model = classifier.official_model(config['model'], n_classes)
    
    # train
    resnet = trainer.ResNetTrainer(
        model = model,
        train_loader = train_loader,
        valid_loader = valid_loader,
        config = config,
        device = device,
        comment = f'resnet{comment}'
    )

    resnet.fit()
    
    
elif sys.argv[1] == 'detector':
    from utils import yolo_utils
    from nets import detector
    config = configs['Detector']

    # load data list
    yolo_lines = utils.read_txt_to_list(config['training_data_list'])

    random.shuffle(yolo_lines)
    cut = int(config['validation_ratio']*len(yolo_lines))
    train_lines = yolo_lines[cut:]
    valid_lines = yolo_lines[:cut]

    # data
    train_data = dataset.YOLODataset(
        yolo_lines=train_lines, config=config, is_train=True
    )

    valid_data = dataset.YOLODataset(
        yolo_lines=valid_lines, config=config, is_train=False
    )

    if config['pretrained']:
        anchors = None
    else:
        # auto anchors
        boxes = yolo_utils.yolo_line_to_boxes(yolo_lines)
        anchors = yolo_utils.generate_anchors(boxes, k=config['anchor_numbers'])
        print(f'Generate {config["anchor_numbers"]} anchors.')

    # model
    model = detector.YOLOv4(config['anchor_numbers'])

    # train
    yolo = trainer.YOLOTrainer(
        model = model,
        train_data = train_data,
        valid_data = valid_data,
        config = config,
        anchors = anchors,
        device = device,
        comment = f'yolo{comment}'
    )

    yolo.fit()