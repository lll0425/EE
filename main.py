import os
import re
import mat73
import torch
from torch.utils.data import DataLoader
#from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
# Model
from utils.tool import train,fixed_seed
from dataset.tool import get_train_val_dataset
from cfg import *
#from model.cnn import Conv_BN_little
from model.resnet import ResNet
#from utils.scheduler import Warmup_MultiStepDecay
import argparse
# from torchsummary import summary
#import torchvision.models as models
#import torchvision
#from torchview import draw_graph
import tensorboard

def main():
    # parser = argparse.ArgumentParser()#傳遞參數的物件
    # parser.add_argument('--use_stft_ratio', type=float, default=1.0)
    # parser.add_argument('--exp_name', help="the name of the experiment", default="", type=str)
    # args = parser.parse_args()
    exp_name = 'a0.01nowindows30423'
    # use_stft_ratio = args.use_stft_ratio

    # #Setting
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # #Set model type
    model = ResNet()
    #print(model)

    # #Set seed
    seed = cfg['seed']
    fixed_seed(seed)

    # #Create train/val set
    print('Start loading data')
    train_data_root = cfg['train_data_root']
    train_ratio = cfg['split_ratio']
    batch_size = cfg['batch_size']
    train_set,val_set = get_train_val_dataset(data_root=train_data_root, train_ratio=train_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers= 4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers= 4,pin_memory=True, drop_last=True)
    # for i ,(anchor_label, positive_label, negative_label) in enumerate(train_loader):
    # # batch_data 是一个包含了一个 batch 的数据
    # # 在这里你可以查看 batch_data 的结构和内容
    #     #print(anchor_label, positive_label, negative_label)
    #     print('anchor_label   = {}'.format(anchor_label[i]))
    #     print('positive_label   = {}'.format(positive_label[i]))
    #     print('negative_label   = {}'.format(negative_label[i]))


    #     break 
    print('End of loading data')
    print('Numbers of training data   = {}'.format(len(train_loader.dataset)))
    print('Numbers of validation data = {}'.format(len(val_loader.dataset)))
    print('Batch Size  = {}'.format(batch_size))

    # #Optimizer
    lr = cfg['lr']
    margin = cfg['margin']
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    #criterion = losses.TripletMarginLoss()       
    criterion  = TripletMarginLoss(margin = margin)
    miner = TripletMarginMiner(margin = margin, type_of_triplets="hard")
    num_epoch = cfg['epoch']

    # #Print training info
    print("Start training!!\n")    
    saved_model_path = cfg["saved_model_path"]
    saved_optimizer_path = cfg["saved_optimizer_path"]

    # Scheduler
    model = model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False, min_lr=5e-5)
    #   半精度
    # for parms in model.parameters():
    #     parms.data = parms.data.half()
    save_path = './save'

    train(model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epoch=num_epoch,
        save_path=save_path,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        miner = miner,
        scheduler = scheduler,
        exp_name=exp_name,
        
        # scheduler = scheduler
        )
    


if __name__ == '__main__':
     main()