import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import sys
from detanet_model import *
from torch_geometric.loader import DataLoader
from e3nn import o3
import os
from torch_geometric.data import Data
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# User-friendly path selection
This is the script based on Detanet
MODEL_PATH = 'path/to/model'
DATASET_PATH = 'path/to/dataset'
LOG_PATH = 'path/to/log'
CHECKPOINT_PATH = 'path/to/checkpoints'

"""
This script trains an EnviroDetaNet model on molecular data.
It loads a dataset, splits it into train/val/test sets, and trains the model.
Key features:
- Uses PyTorch Geometric for graph neural networks
- Implements a custom Trainer class for model training
- Saves checkpoints and logs during training
- Allows customization of model hyperparameters and training settings

Usage:
1. Set the paths for model, dataset, logs, and checkpoints
2. Adjust hyperparameters as needed
3. Run the script to train the model
"""

model=EnviroDetaNet(num_features=512,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=35,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out=None,
                 summation=False,
                 norm=False,
                 out_type='scalar',
                 grad_type='Hi',
                 device=torch.device('cuda'))
print(model)

datasets=torch.load(DATASET_PATH)
print(datasets[0])


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, loss_function=l2loss, device=torch.device('cuda'),
             optimizer='Adam_amsgrad', lr=5e-4, weight_decay=0, max_grad_norm=1.0,
             log_file=LOG_PATH):
        self.model = model
        self.train_data = train_loader
        self.val_data = val_loader
        self.max_grad_norm = max_grad_norm
        self.loss_function = loss_function
        self.device = device
        self.opts = {
            'AdamW': torch.optim.AdamW(self.model.parameters(), lr=lr, amsgrad=False, weight_decay=weight_decay),
            'AdamW_amsgrad': torch.optim.AdamW(self.model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay),
            'Adam': torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=False, weight_decay=weight_decay),
            'Adam_amsgrad': torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay),
            'Adadelta': torch.optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'RMSprop': torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'SGD': torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        }
        self.optimizer = self.opts[optimizer]
        self.step = -1
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.log_file = log_file
        with open(self.log_file, 'w') as file:
            file.write('Training Log\n')
            file.write('=================\n')

    def log(self, message):
        with open(self.log_file, 'a') as file:
            file.write(message + "\n")

    def train(self, num_epoch, targ, stop_loss=1e-8, element=None, q9=None, loss_area=[100000, 1e-8, 1e8],
              val_per_train=10, view_data=False, print_per_epoch=10, data_scale=1):
        self.model.train()
        len_train = len(self.train_data)
        epoch = num_epoch
        for i in range(epoch):
            val_datas = iter(self.val_data)
            for j, batch in enumerate(self.train_data):
                self.step += 1
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                out = self.model(pos=batch.pos.clone().detach().to(self.device).to(torch.float32),
                                z=batch.z.to(self.device), batch=batch.batch.to(self.device),
                                edge_index=batch.edge_index.to(self.device),smiles=batch.smiles)

                target = batch[targ].to(self.device) * data_scale
                loss = self.loss_function(out.reshape(target.shape), target)
                if self.step < loss_area[0] or (loss_area[1] < loss.item() < loss_area[2]):
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                if self.step % val_per_train == 0 and self.val_data is not None:
                    val_batch = next(val_datas)
                    val_target = val_batch[targ].to(self.device).reshape(-1) * data_scale
                    val_out = self.model(pos=torch.tensor(val_batch.pos, device=self.device, dtype=torch.float32),
                                         z=val_batch.z.to(self.device), batch=val_batch.batch.to(self.device),
                                         edge_index=val_batch.edge_index.to(self.device),smiles=val_batch.smiles).reshape(val_target.shape)
                    val_loss = self.loss_function(val_out, val_target).item()
                    val_mae = l1loss(val_out, val_target).item()
                    val_R2 = R2(val_out, val_target).item()
                    if self.step % print_per_epoch == 0:
                        log_message = ('Epoch[{}/{}],loss:{:.8f},val_loss:{:.8f},val_mae:{:.8f},val_R2:{:.8f}'
                                       .format(self.step, num_epoch * len_train, loss.item(), val_loss, val_mae, val_R2))
                        self.log(log_message)

                        if view_data:
                            self.log('valout:{:.8f},valtarget:{:.8f}'.format(val_out.flatten()[0].item(),
                                                                             val_target.flatten()[0].item()))
                    assert (loss > stop_loss) or (val_loss > stop_loss), 'Training and prediction Loss is less than cut-off Loss, so training stops'
                elif self.step % print_per_epoch == 0:
                    self.log('Epoch[{}/{}],loss:{:.8f}'.format(self.step, num_epoch * len_train, loss.item()))
                if self.step % 5000 == 0:
                    self.save_param(os.path.join(CHECKPOINT_PATH, f'{self.step}.pth'))
    def load_state_and_optimizer(self,state_path=None,optimizer_path=None):
        if state_path is not None:
            state_dict=torch.load(state_path)
            self.model.load_state_dict(state_dict)
        if optimizer_path is not None:
            self.optimizer=torch.load(optimizer_path)

    def save_param(self,path):
        torch.save(self.model.state_dict(),path)

    def save_model(self,path):
        torch.save(self.model, path)

    def save_opt(self,path):
        torch.save(self.optimizer,path)

    def params(self):
        return self.model.state_dict()


    def load_state_and_optimizer(self,state_path=None,optimizer_path=None):
        if state_path is not None:
            state_dict=torch.load(state_path)
            self.model.load_state_dict(state_dict)
        if optimizer_path is not None:
            self.optimizer=torch.load(optimizer_path)

    def save_param(self,path):
        torch.save(self.model.state_dict(),path)

    def save_model(self,path):
        torch.save(self.model, path)

    def save_opt(self,path):
        torch.save(self.optimizer,path)

    def params(self):
        return self.model.state_dict()
    
import random
from torch_geometric.data import Data

def random_split_datasets(datasets, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    random.shuffle(datasets)  
    
    total_size = len(datasets)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for i, data in enumerate(datasets):
        zerr = 0
        targ = 'Hi'
        data_ = Data(pos=data.pos, z=data.z, edge_index=data.edge_index, targ=data[targ], smiles=data.smile)
        
        if zerr == 0:
            if i < train_size:
                train_datasets.append(data_)
            elif i < train_size + val_size:
                val_datasets.append(data_)
            else:
                test_datasets.append(data_)
        else:
            print('error', data.smile)
    
    return train_datasets, val_datasets, test_datasets

train_datasets, val_datasets, test_datasets = random_split_datasets(datasets)


bathes=32
trainloader=DataLoader(train_datasets,batch_size=bathes,shuffle=True)
valloader=DataLoader(val_datasets,batch_size=bathes,shuffle=True)

device=torch.device('cuda')
dtype=torch.float32
model=model.to(dtype)
model=model.to(device)

print('start!!')
trainer = Trainer(model, train_loader=trainloader, val_loader=valloader, loss_function=l2loss, lr=5e-4, weight_decay=0, optimizer='AdamW_amsgrad', max_grad_norm=1.0)
trainer.save_param(os.path.join(MODEL_PATH, 'Hi2.pth'))
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'Hi2.pth')))
trainer.train(num_epoch=200, stop_loss=0, targ='targ', val_per_train=30, view_data=False, print_per_epoch=10, loss_area=[1e8,1e-8,1e8])
torch.cuda.empty_cache()
print(trainer.model.state_dict())
