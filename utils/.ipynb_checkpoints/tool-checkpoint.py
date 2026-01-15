#tool.py
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import miners,  losses
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import numpy as np
import random
import os
import time
# from torchmetrics.classification  import BinaryF1Score
# from torchmetrics.functional import precision_recall
from sklearn.metrics import precision_recall_fscore_support ,f1_score
from sklearn.preprocessing import OneHotEncoder
#from classifier import TransmitterClassifier, ReceiverClassifier, GradientReversalLayer

def fixed_seed(myseed:int):
    """Initial random seed
    Args:
        myseed: the seed for initial
    """
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def custom_lr_lambda(epoch):
    if epoch >= 300:
        return 0.01
    elif epoch >= 100:
        return 0.1
    else:
        return 1.0


def train(model, train_loader, val_loader, num_epoch:int, save_path:str, device, criterion, optimizer, miner,scheduler, exp_name=""):#every_step_update=1):
    start_train = time.time()

    #########################################
    #miner = miners.TripletMarginMiner(margin = 0.2, type_of_triplets="hard")
    # loss_func = losses.TripletMarginLoss()
    ##########################################

    if exp_name == "":
        writer = SummaryWriter()
    else:
        path = f"./runs/"
        os.makedirs(path,exist_ok=True)
        writer = SummaryWriter(os.path.join(path, exp_name))
    # #Create directory for saving model.
    os.makedirs(save_path,exist_ok=True)
    # scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_lambda)

    # Best NME loss and epoch for save best .pt
    best_val_epoch = 0
    best_loss=999

    for epoch in range((num_epoch)):
        print(f'epoch = {epoch}')
        start_time = time.time()
        writer.add_scalar(tag="train/learning_rate",
                            scalar_value=float(optimizer.param_groups[0]['lr']),
                            global_step=epoch)
        print(f"Current Learning rate = {optimizer.param_groups[0]['lr']:.6f}")
        model.train()
        train_loss = 0.0
        num_train_triplet = 0
        ###############################################################
        for i ,(data, tx_labels) in enumerate(tqdm(train_loader)):
            anchor_CIS, anchor_tx = data, tx_labels
            anchor_CIS, anchor_tx = anchor_CIS.to(device), anchor_tx.to(device)

            # anchor_tx = torch.squeeze(anchor_tx, dim=1)
            # anchor_tx = torch.argmax(anchor_tx, dim=1)
            out_anchor = model(anchor_CIS)

            #neuron_outputs = out_anchor[0].tolist()

            # # 打印每个神经元的输出
            # for i, output_value in enumerate(neuron_outputs):
            #     print(f"Neuron {i+1} Output: {output_value}")

            hard_pairs = miner(out_anchor, anchor_tx)
            num_triplet = hard_pairs[0].size(0)
            # if i == 0:
            #     print(anchor_labels)
            #     print(hard_pairs[1][1])
            loss = criterion(out_anchor, anchor_tx, hard_pairs)
            #print(loss.item())
            # if i == 2:
            #     print(loss.item())
            #     print(loss)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_train_triplet += num_triplet

        train_loss /= len(train_loader)
        writer.add_scalar(tag="train/loss",
                        scalar_value=float(train_loss),
                        global_step=epoch)
        writer.add_scalar(tag="triplet/num_train_triplet",
                        scalar_value=float(num_train_triplet),
                        global_step=epoch)
        

        with torch.no_grad():
            eval_loss = 0
            num_val_triplet = 0
            model.eval()
            for i ,(data, tx_labels) in enumerate(tqdm(val_loader)):
                anchor_CIS, anchor_tx = data, tx_labels

                # anchor_tx = torch.squeeze(anchor_tx, dim=1)
                # anchor_tx = torch.argmax(anchor_tx, dim=1)


                anchor_CIS = anchor_CIS.to(device)
                out_anchor = model(anchor_CIS)
                hard_pairs = miner(out_anchor, anchor_tx)
                loss = criterion(out_anchor, anchor_tx)
                eval_loss += loss.item()
                eval_num_triplet = hard_pairs[0].size(0)
                num_val_triplet += eval_num_triplet

        eval_loss /= len(val_loader)
        writer.add_scalar(tag="val/loss",
                            scalar_value=float(eval_loss),
                            global_step=epoch)
        writer.add_scalar(tag="triplet/num_val_triplet",
                            scalar_value=float(num_val_triplet),
                            global_step=epoch)
        scheduler.step(num_val_triplet)
        end_time = time.time()
        elp_time = end_time - start_time

        print('='*24)
        print('time = {} MIN {:.1f} SEC, total time = {} Min {:.1f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        formatted_str = "{: <20} : {:.6f}"
        print(formatted_str.format('Training loss', train_loss))

        print(formatted_str.format('Validation  loss', eval_loss))
        print('='*24 + '\n')
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(),os.path.join(save_path,'best.pt'))

        # Save model, scheduler and optimizer
        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_{epoch}.pt'))
        #scheduler.step()


    print("End of training !!!")
    print(f"Best val loss {best_loss:.6f} on epoch {best_val_epoch}")

    # writer.add_hparams(train_hyp, metric_result)

    writer.close()

# def train(model, train_loader, val_loader, num_epoch:int, save_path:str, device, criterion, optimizer, miner,scheduler, exp_name=""):#every_step_update=1):
#     start_train = time.time()

#     #########################################
#     #miner = miners.TripletMarginMiner(margin = 0.2, type_of_triplets="hard")
#     # loss_func = losses.TripletMarginLoss()
#     ##########################################

#     if exp_name == "":
#         writer = SummaryWriter()
#     else:
#         path = f"./runs/"
#         os.makedirs(path,exist_ok=True)
#         writer = SummaryWriter(os.path.join(path, exp_name))
#     # #Create directory for saving model.
#     os.makedirs(save_path,exist_ok=True)
#     # scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_lambda)

#     # Best NME loss and epoch for save best .pt
#     best_val_epoch = 0
#     best_loss=999

#     for epoch in range((num_epoch)):
#         print(f'epoch = {epoch}')
#         start_time = time.time()
#         writer.add_scalar(tag="train/learning_rate",
#                             scalar_value=float(optimizer.param_groups[0]['lr']),
#                             global_step=epoch)
#         print(f"Current Learning rate = {optimizer.param_groups[0]['lr']:.6f}")
#         model.train()
#         train_loss = 0.0
#         num_train_triplet = 0
#         ###############################################################
#         for i ,(data, labels) in enumerate(tqdm(train_loader)):
#             anchor_CIS, anchor_labels = data, labels
#             anchor_CIS, anchor_labels = anchor_CIS.to(device), anchor_labels.to(device)
            
#             out_anchor = model(anchor_CIS)

#             #neuron_outputs = out_anchor[0].tolist()

#             # # 打印每个神经元的输出
#             # for i, output_value in enumerate(neuron_outputs):
#             #     print(f"Neuron {i+1} Output: {output_value}")

#             hard_pairs = miner(out_anchor, anchor_labels)
#             num_triplet = hard_pairs[0].size(0)
#             # if i == 0:
#             #     print(anchor_labels)
#             #     print(hard_pairs[1][1])
#             loss = criterion(out_anchor, anchor_labels, hard_pairs)
#             #print(loss.item())
#             # if i == 2:
#             #     print(loss.item())
#             #     print(loss)
                
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             num_train_triplet += num_triplet

#         train_loss /= len(train_loader)
#         writer.add_scalar(tag="train/loss",
#                         scalar_value=float(train_loss),
#                         global_step=epoch)
#         writer.add_scalar(tag="triplet/num_train_triplet",
#                         scalar_value=float(num_train_triplet),
#                         global_step=epoch)
        

#         with torch.no_grad():
#             eval_loss = 0
#             num_val_triplet = 0
#             model.eval()
#             for i ,(data, labels) in enumerate(tqdm(val_loader)):
#                 anchor_CIS, anchor_labels = data, labels
#                 anchor_CIS = anchor_CIS.to(device)
#                 out_anchor = model(anchor_CIS)
#                 hard_pairs = miner(out_anchor, anchor_labels)
#                 loss = criterion(out_anchor, anchor_labels)
#                 eval_loss += loss.item()
#                 eval_num_triplet = hard_pairs[0].size(0)
#                 num_val_triplet += eval_num_triplet

#         eval_loss /= len(val_loader)
#         writer.add_scalar(tag="val/loss",
#                             scalar_value=float(eval_loss),
#                             global_step=epoch)
#         writer.add_scalar(tag="triplet/num_val_triplet",
#                             scalar_value=float(num_val_triplet),
#                             global_step=epoch)
#         scheduler.step(num_val_triplet)
#         end_time = time.time()
#         elp_time = end_time - start_time

#         print('='*24)
#         print('time = {} MIN {:.1f} SEC, total time = {} Min {:.1f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
#         formatted_str = "{: <20} : {:.6f}"
#         print(formatted_str.format('Training loss', train_loss))

#         print(formatted_str.format('Validation  loss', eval_loss))
#         print('='*24 + '\n')
#         if eval_loss < best_loss:
#             best_loss = eval_loss
#             best_val_epoch = epoch
#             torch.save(model.state_dict(),os.path.join(save_path,'best.pt'))

#         # Save model, scheduler and optimizer
#         torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pt'))
#         torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_{epoch}.pt'))
#         #scheduler.step()


#     print("End of training !!!")
#     print(f"Best val loss {best_loss:.6f} on epoch {best_val_epoch}")

#     # writer.add_hparams(train_hyp, metric_result)

#     writer.close()