from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from torch import nn, optim
from PIL import Image
import torchvision.transforms as tsf
import numpy as np
import torch
import time
import os

from utils import utils, torch_utils, yolo_utils, dataset


class ResNetTrainer:
    
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            config,
            device = 'cuda',
            comment: str = 'resnet'
        ):
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epoch = config['epoch']
        
        self.model = model.to(self.device)
        # model weight init
        if config['pretrained'] is None:
            self.model = self.model.apply(torch_utils.kaiming_init)
        else:
            checkpoint = torch.load(config['pretrained'], map_location=device)
            self.model.load_state_dict(checkpoint)
            print(f'Load weight file {os.path.basename(config["pretrained"])} successfully.')
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.BCELoss(reduction='mean')
        
        self.keep_pth = config['keep_pth']
        assert self.keep_pth in ['all', 'best', 'false']
        
        self.log_path = f'logs/{datetime.now().strftime("%y-%m-%d-%H%M")}-{comment}'
        self.summary = {
            'Plan Name': f'{os.path.basename(self.log_path)}',
            'Model': 'single head yolov4',
            'Input Dimension [C, H, W]': [self.model.in_channels, *config['input_size'][::-1]],
            'Batch Size': config['training_batch_size'],
            'Initial Learning Rate': config['learning_rate'],
            'Pretrained': config['pretrained']
        }
        self.pad = len(str(self.epoch))
        
    def _check_answers(self, outputs, targets):
        outputs = torch.where(outputs>0.5, 1, 0)
        corrects = torch.all(outputs==targets, dim=1)
        return corrects.tolist()
        
    def _train(self):
        self.model.train()  # training mode
        losses, corrects = [], []

        # set a batch with tqdm
        batch_iter = tqdm(
            iterable = self.train_loader,
            desc = '[Train]',
            total = len(self.train_loader),
            ncols = 80,
            postfix = dict
        )
        
        # strat batches
        for x, y in batch_iter:
            inputs, targets = x.to(self.device), y.to(self.device)  # set device
            self.optimizer.zero_grad()  # set grad params zero

            outputs = self.model(inputs)  # forward propagation
            loss = self.criterion(outputs, targets)  # calculate loss
            
            # get results
            losses.append(loss.item())
            corrects += self._check_answers(outputs, targets)
            
            # backpropagation
            loss.backward()
            self.optimizer.step()  # update weights
            
            # show info
            batch_iter.set_postfix(**{
                'loss': sum(losses) / len(losses),
                'acc': sum(corrects) / len(corrects)
            })
        
        self.train_losses.append(sum(losses) / len(losses))
        self.train_acc.append(sum(corrects) / len(corrects))
        batch_iter.close()

    def _valid(self):
        self.model.eval()  # evaluation mode
        with torch.no_grad():
            losses, corrects = [], []

            # set a batch with tqdm
            batch_iter = tqdm(
                iterable = self.valid_loader,
                desc = '[Valid]',
                total = len(self.valid_loader),
                ncols = 80,
                postfix = dict
            )

            for x, y in batch_iter:
                inputs, targets = x.to(self.device), y.to(self.device)  # set device

                outputs = self.model(inputs)  # forward propagation
                loss = self.criterion(outputs, targets)  # calculate loss
                losses.append(loss.item())
                corrects += self._check_answers(outputs, targets)

                # show info
                batch_iter.set_postfix(**{
                    'loss': sum(losses) / len(losses),
                    'acc': sum(corrects) / len(corrects)
                })

            self.valid_losses.append(sum(losses) / len(losses))
            self.valid_acc.append(sum(corrects) / len(corrects))
            batch_iter.close()
    
    def fit(self):
        t0 = time.time()

        # init
        utils.create_dir(self.log_path)
        epoch, self.train_losses, self.valid_losses = [], [], []
        self.train_acc, self.valid_acc = [], []
        fields = [
            'Epoch', 'Training Loss', 'Validation Loss', 
            'Training Accuracy', 'Validation Accuracy'
        ]
        best_epoch, best_loss, best_acc = 0, 1e+6, 0
        
        if self.keep_pth == 'all':
            utils.create_dir(f'{self.log_path}/pth', check_overwrite=False)
        elif self.keep_pth=='best':
            best_pth = None
        
        # training
        for i in range(self.epoch):
            i += 1
            print(f'---Epoch {i}/{self.epoch}---')
            epoch.append(i)
            self._train()
            self._valid()
            
            if self.valid_losses[-1] < best_loss:
                best_epoch = i
                best_loss = self.valid_losses[-1]
                
            # save model
            if self.keep_pth == 'all':
                torch.save(
                    self.model.state_dict(), 
                    f'{self.log_path}/pth/epoch{i:0>{self.pad}d}.pth'
                )
                
            elif self.keep_pth=='best' and best_epoch==i:
                if best_pth:
                    os.remove(best_pth)
                best_pth = f'{self.log_path}/epoch{i:0>{self.pad}d}.pth'
                torch.save(self.model.state_dict(), best_pth)
                    
            # write csv logs
            data = zip(
                epoch, self.train_losses, self.valid_losses,
                self.train_acc, self.valid_acc
            )
            utils.keep_training_log(self.log_path, fields, data)
                
            best_loss = min(self.valid_losses)
            best_epoch = self.valid_losses.index(best_loss) + 1
            best_acc = self.valid_acc[best_epoch-1]
            
            # write summary
            self.summary.update({
                'End Epoch': i,
                'Total Training Time:': f'{(time.time()-t0)/3600:.1f} hours',
                'Best Epoch:': best_epoch,
                'Best Validation Loss:': f'{best_loss:.4f}',
                'Accuracy of the Best Epoch': f'{100*best_acc:.1f} %'
            })
            utils.keep_summary_to_txt(self.log_path, self.summary)
            
        print(f'Completed!')


class UNetTrainer:
    
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            config,
            quick_result: list = None,
            device = 'cuda',
            comment: str = 'unet'
        ):
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epoch = config['epoch']
        
        self.quick_result = quick_result
        if quick_result:
            self.size = config['input_size']
        
        self.model = model.to(self.device)
        
        # model weight init
        if config['pretrained'] is None:
            self.model = self.model.apply(torch_utils.kaiming_init)
        else:
            checkpoint = torch.load(config['pretrained'], map_location=device)
            self.model.load_state_dict(checkpoint)
            print(f'Load weight file {os.path.basename(config["pretrained"])} successfully.')
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        self.keep_pth = config['keep_pth']
        assert self.keep_pth in ['all', 'best', 'false']
        
        self.log_path = f'logs/{datetime.now().strftime("%y-%m-%d-%H%M")}-{comment}'
        self.summary = {
            'Plan Name': f'{os.path.basename(self.log_path)}',
            'Model': config['model'],
            'Input Dimension [C, H, W]': [self.model.in_channels, *config['input_size'][::-1]],
            'Batch Size': config['training_batch_size'],
            'Initial Learning Rate': config['learning_rate'],
            'Validation Ratio': config['validation_ratio'],
            'Pretrained': config['pretrained']
        }
        self.pad = len(str(self.epoch))
        
    def _output_quick_result(self, epoch):
        self.model.eval()  # evaluation mode
        with torch.no_grad():
            for img_name, ts_img in zip(self.img_names, self.ts_imgs):
                output = self.model(ts_img)  # forward propagation
                output = output.squeeze()
                output = torch.where(output>0, 1, 0)   # get classes for pixels
                output = Image.fromarray(np.uint8(output.cpu().data.numpy()*255))  # tensor to numpy to pil img
                output.save(f'{self.log_path}/quick_results/{img_name}_epoch{epoch:0>{self.pad}d}.jpg')
        
    def _train(self):
        self.model.train()  # training mode
        losses = []  # loss container

        # set a batch with tqdm
        batch_iter = tqdm(
            iterable = self.train_loader,
            desc = '[Train]',
            total = len(self.train_loader),
            ncols = 80,
            postfix = dict
        )
        
        # strat batches
        for x, y in batch_iter:
            inputs, targets = x.to(self.device), y.to(self.device)  # set device

            self.optimizer.zero_grad()  # set grad params zero

            outputs = self.model(inputs)  # forward propagation
            loss = self.criterion(outputs, targets)  # calculate loss
            losses.append(loss.item())
            
            # backpropagation
            loss.backward()
            self.optimizer.step()  # update weights
            
            # show info
            batch_iter.set_postfix(**{
                'loss': sum(losses) / len(losses)
            })
        
        self.train_losses.append(sum(losses) / len(losses))
        batch_iter.close()

    def _valid(self):
        self.model.eval()  # evaluation mode
        with torch.no_grad():
            losses = []  # loss container

            # set a batch with tqdm
            batch_iter = tqdm(
                iterable = self.valid_loader,
                desc = '[Valid]',
                total = len(self.valid_loader),
                ncols = 80,
                postfix = dict
            )

            for x, y in batch_iter:
                inputs, targets = x.to(self.device), y.to(self.device)  # set device

                outputs = self.model(inputs)  # forward propagation
                loss = self.criterion(outputs, targets)  # calculate loss
                losses.append(loss.item())

                # show info
                batch_iter.set_postfix(**{
                    'loss': sum(losses) / len(losses)
                })

            self.valid_losses.append(sum(losses) / len(losses))
            batch_iter.close()
    
    def fit(self):
        t0 = time.time()

        # init
        utils.create_dir(self.log_path)
        epoch, self.train_losses, self.valid_losses = [], [], []
        fields = ['Epoch', 'Training Loss', 'Validation Loss']
        best_epoch, best_loss = 0, 1e+6
        
        if self.keep_pth == 'all':
            utils.create_dir(f'{self.log_path}/pth', check_overwrite=False)
        elif self.keep_pth=='best':
            best_pth = None
        
        # prepare quick samples
        if self.quick_result:
            utils.create_dir(f'{self.log_path}/quick_results', check_overwrite=False)
            self.img_names, self.ts_imgs = [], []
            for i, path in enumerate(self.quick_result):
                img = Image.open(path).convert('RGB')
                name = f'test-{i:0>{self.pad}d}'
                img.save(f'{self.log_path}/quick_results/{name}.jpg')
                self.img_names.append(name)
                ts_img = torch_utils.cook_input(img, self.size, device=self.device) 
                self.ts_imgs.append(ts_img)
        
        # training
        for i in range(self.epoch):
            i += 1
            print(f'---Epoch {i}/{self.epoch}---')
            epoch.append(i)
            self._train()
            self._valid()
            
            if self.valid_losses[-1] < best_loss:
                best_epoch = i
                best_loss = self.valid_losses[-1]
                
            # save model
            if self.keep_pth == 'all':
                torch.save(
                    self.model.state_dict(), 
                    f'{self.log_path}/pth/epoch{i:0>{self.pad}d}.pth'
                )
                
            elif self.keep_pth=='best' and best_epoch==i:
                if best_pth:
                    os.remove(best_pth)
                best_pth = f'{self.log_path}/epoch{i:0>{self.pad}d}.pth'
                torch.save(self.model.state_dict(), best_pth)
                    
            # write csv logs
            data = zip(epoch, self.train_losses, self.valid_losses)
            utils.keep_training_log(self.log_path, fields, data)
            
            # quick result
            if self.quick_result:
                self._output_quick_result(i)
                
            best_loss = min(self.valid_losses)
            best_epoch = self.valid_losses.index(best_loss) + 1
            
            # write summary
            self.summary.update({
                'End Epoch': i,
                'Total Training Time:': f'{(time.time()-t0)/3600:.1f} hours',
                'Best Epoch:': best_epoch,
                'Best Validation Loss:': f'{best_loss:.4f}'
            })
            utils.keep_summary_to_txt(self.log_path, self.summary)
            
        print(f'Completed!')


class YOLOTrainer:
    
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: torch.utils.data.Dataset,
            valid_data: torch.utils.data.Dataset,
            config,
            anchors = None,
            device = 'cuda',
            comment: str = 'yolo'
        ):
        self.device = device
        self.train_data = train_data
        self.valid_data = valid_data
        self.freeze_bs = config['freeze_batch_size']
        self.unfreeze_bs = config['unfreeze_batch_size']
        self.num_workers = config['num_workers']
        self.epoch = config['total_epoch']
        self.unfreeze_epoch = config['unfreeze_epoch']
        self.init_lr = config['learning_rate']
        self.lr_decay = config['lr_decay']
        self.model = model.to(device)
           
        # model weight init
        if config['pretrained'] is None:
            self.anchors = anchors.to(device)
            self.model = self.model.apply(torch_utils.kaiming_init)
            print('Initialize the model with Kaiming method.')
        else:
            checkpoint = torch.load(config['pretrained'], map_location=device)
            self.anchors = checkpoint['anchors']
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f'Load weight file {os.path.basename(config["pretrained"])} successfully.')
        
        self.criterion = yolo_utils.YOLOLoss(self.anchors)
        
        self.keep_pth = config['keep_pth']
        assert self.keep_pth in ['all', 'best', 'false']
        
        self.log_path = f'logs/{datetime.now().strftime("%y-%m-%d-%H%M")}-{comment}'
        self.summary = {
            'Plan Name': f'{os.path.basename(self.log_path)}',
            'Model': 'single-head-yolov4',
            'Input Dimension [C, H, W]': [self.model.in_channels, *config['input_size'][::-1]],
            'Freeze Batch Size': config['freeze_batch_size'],
            'Unfreeze Batch Size': config['unfreeze_batch_size'],
            'Base Learning Rate': config['learning_rate'],
            'Pretrained': config['pretrained'],
            'Anchors': self.anchors.tolist(),
            'Unfreeze Epoch at': config['unfreeze_epoch']
        }
        self.pad = len(str(self.epoch))
    
    def _set_freeze_state(self, freeze:bool):
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.lr_decay
        )
        
        batch_size = self.freeze_bs if freeze else self.unfreeze_bs
        self.train_loader = DataLoader(
            self.train_data,
            batch_size = batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            collate_fn = dataset.yolo_collator
        )
        self.valid_loader = DataLoader(
            self.valid_data,
            batch_size = batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            collate_fn = dataset.yolo_collator
        )
        
        state = 'freezed' if freeze else 'unfreezed'
        print(f'Step backbone of the yolo {state}.')
        
    def _train(self):
        self.model.train()  # training mode
        self.lr += self.scheduler.get_last_lr()
        losses, ciou_losses, obj_losses = [], [], []

        # set a batch with tqdm
        batch_iter = tqdm(
            iterable = self.train_loader,
            desc = '[Train]',
            total = len(self.train_loader),
            ncols = 90,
            postfix = dict
        )
        
        # strat batches
        for imgs, labels, boxes in batch_iter:
            inputs = imgs.to(self.device)
            labels = [t if t==[] else t.to(self.device) for t in labels]
            boxes = [t if t==[] else t.to(self.device) for t in boxes]
            targets = labels, boxes
            
            self.optimizer.zero_grad()  # set grad params zero

            outputs = self.model(inputs)  # forward propagation
            ciou_loss, obj_loss = self.criterion(outputs, targets)  # calculate loss
            loss = ciou_loss + obj_loss
            
            # get results
            ciou_losses.append(ciou_loss.item())
            obj_losses.append(obj_loss.item())
            losses.append(loss.item())
            
            # backpropagation
            loss.backward()
            self.optimizer.step()  # update weights
            
            # show info
            batch_iter.set_postfix(**{
                'box_loss': sum(ciou_losses) / len(ciou_losses),
                'grid_loss': sum(obj_losses) / len(obj_losses)
            })
        
        self.train_losses.append(sum(losses) / len(losses))
        self.train_ciou.append(sum(ciou_losses) / len(ciou_losses))
        self.train_grid.append(sum(obj_losses) / len(obj_losses))
        self.scheduler.step()
        batch_iter.close()

    def _valid(self):
        self.model.eval()  # evaluation mode
        with torch.no_grad():
            losses, ciou_losses, obj_losses = [], [], []

            # set a batch with tqdm
            batch_iter = tqdm(
                iterable = self.valid_loader,
                desc = '[Valid]',
                total = len(self.valid_loader),
                ncols = 90,
                postfix = dict
            )

            for imgs, labels, boxes in batch_iter:
                inputs = imgs.to(self.device)
                labels = [t if t==[] else t.to(self.device) for t in labels]
                boxes = [t if t==[] else t.to(self.device) for t in boxes]
                targets = labels, boxes

                outputs = self.model(inputs)  # forward propagation
                ciou_loss, obj_loss = self.criterion(outputs, targets)  # calculate loss
                loss = ciou_loss + obj_loss
                
                ciou_losses.append(ciou_loss.item())
                obj_losses.append(obj_loss.item())
                losses.append(loss.item())

                # show info
                batch_iter.set_postfix(**{
                    'box_loss': sum(ciou_losses) / len(ciou_losses),
                    'grid_loss': sum(obj_losses) / len(obj_losses)
                })

            self.valid_losses.append(sum(losses) / len(losses))
            self.valid_ciou.append(sum(ciou_losses) / len(ciou_losses))
            self.valid_grid.append(sum(obj_losses) / len(obj_losses))
            batch_iter.close()
    
    def _save_state_dict(self, path):
        yolo_dict = {
            'state_dict': self.model.state_dict(),
            'anchors': self.anchors
        }
        torch.save(yolo_dict, path)
    
    def fit(self):
        t0 = time.time()

        # init
        utils.create_dir(self.log_path)
        epoch, self.train_losses, self.valid_losses = [], [], []
        self.lr, self.train_ciou, self.valid_ciou = [], [], []
        self.train_grid, self.valid_grid = [], []
        fields = [
            'Epoch', 'Training Loss', 'Validation Loss', 
            'Training CIoU Loss', 'Validation CIoU Loss',
            'Training Grid Loss', 'Validation Grid Loss',
            'Learning Rate'
        ]
        best_epoch, best_loss = 0, 1e+6
        
        if self.keep_pth == 'all':
            utils.create_dir(f'{self.log_path}/pth', check_overwrite=False)
        elif self.keep_pth=='best':
            best_pth = None
        
        if self.unfreeze_epoch > 1:
            self._set_freeze_state(freeze=True)
        
        # training
        for i in range(self.epoch):
            i += 1
            print(f'---Epoch {i}/{self.epoch}---')
            epoch.append(i)
            
            if i == self.unfreeze_epoch:
                self._set_freeze_state(freeze=False)
            
            self._train()
            self._valid()
            
            if self.valid_losses[-1] < best_loss:
                best_epoch = i
                best_loss = self.valid_losses[-1]
                
            # save model
            if self.keep_pth == 'all':
                self._save_state_dict(f'{self.log_path}/pth/epoch{i:0>{self.pad}d}.pth')
                
            elif self.keep_pth=='best' and best_epoch==i:
                if best_pth:
                    os.remove(best_pth)
                best_pth = f'{self.log_path}/epoch{i:0>{self.pad}d}.pth'
                self._save_state_dict(best_pth)
                    
            # write csv logs
            data = zip(
                epoch, self.train_losses, self.valid_losses,
                self.train_ciou, self.valid_ciou,
                self.train_grid, self.valid_grid, self.lr
            )
            utils.keep_training_log(self.log_path, fields, data)
                
            best_loss = min(self.valid_losses)
            best_epoch = self.valid_losses.index(best_loss) + 1
            
            # write summary
            self.summary.update({
                'End Epoch': i,
                'End Learning Rate': self.lr[-1],
                'Total Training Time:': f'{(time.time()-t0)/3600:.1f} hours',
                'Best Epoch:': best_epoch,
                'Best Validation Loss:': f'{best_loss:.6f}',
            })
            utils.keep_summary_to_txt(self.log_path, self.summary)
            
        print(f'Completed!')