import numpy as np
from glob import glob
import sys
sys.path.append('D:/午/EXTRACTOR')
from cfg import *
import os
import pickle
import torch
from dataset.CISDataset import CISDataset, RxAgnosticCISDataset
from torch.utils.data import random_split
from tqdm import tqdm
import random
from collections import defaultdict
def get_train_val_dataset(data_root: str, train_ratio: float, val_ratio: float):
    dataset = RxAgnosticCISDataset(data_root)
    print(f"Loaded dataset with {len(dataset)} samples.")

    train_data, val_data, test_data = [], [], []

    total_samples = len(dataset)
    num_train = int(total_samples * train_ratio)
    num_val = int(total_samples * val_ratio)
    num_test = total_samples - num_train - num_val

    train_data, remaining_data = random_split(dataset, [num_train, total_samples - num_train])
    val_data, test_data = random_split(remaining_data, [num_val, num_test])

    print(f"Training data: {len(train_data)}, Validation data: {len(val_data)}, Testing data: {len(test_data)}")
    return train_data, val_data, test_data