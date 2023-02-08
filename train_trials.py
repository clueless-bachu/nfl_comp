import os
import glob
import numpy as np
import pandas as pd
import random
import math
import gc
import cv2
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import re
tqdm.pandas()
from sklearn.metrics import matthews_corrcoef
from helper import *
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
# import timm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from itertools import cycle


class MyDataset(Dataset):
    def __init__(self, df, feature_cols = ['rel_pos_x',
       'rel_pos_y', 'rel_pos_mag', 'rel_pos_ori', 'rel_speed_x', 'rel_speed_y',
       'rel_speed_mag', 'rel_speed_ori', 'rel_acceleration_x',
       'rel_acceleration_y', 'rel_acceleration_mag', 'rel_acceleration_ori',
       'G_flug']):
        
        self.df = df
        self.features = feature_cols
        
    def __len__(self):
        return len(self.df)
   
    def __getitem__(self, idx):
        window = 24
        frames_to_skip = 4
        
        row = self.df.loc[idx]
        mid_frame = row['frame']
        
        label = row['contact']
        
#         print(f'row data: {row}')
#         print(f'mid_frame = {mid_frame}, label = {label}')
        imgs = []
        for view in ['Endzone', 'Sideline']:
            video = row['game_play'] +  f'_{view}.mp4'
            frames = [mid_frame - window + i for i in range(0, 2*window+1, frames_to_skip)]
            
            ## this 4 frame skip is a hyperparameter that corresponds to 15 fps
            bbox_col = 'bbox_endzone' if view == 'Endzone' else 'bbox_sideline'
            bboxes = row[bbox_col][::frames_to_skip].astype(np.int32) 
            
            ## incase the person cannot be viewed from a particular location, bbox.sum is 0
            if bboxes.sum() <= 0:
                imgs += [np.zeros((256, 256), dtype=np.float32)]*len(frames)
                continue
            
            for i, frame in enumerate(frames):
                img_new = np.zeros((256, 256), dtype=np.float32)
                cx, cy = bboxes[i]
                path = f'./work/train_frames/{video}_{frame:04d}.jpg'
                if os.path.isfile(path):
                    img_new = np.zeros((256, 256), dtype=np.float32)
                    if view == 'Endzone':
                        img = cv2.imread(path, 0)[cy-76:cy+180, cx-128:cx+128].copy()
                        img_new[:img.shape[0], :img.shape[1]] = img
                    else:
                        img = cv2.imread(path, 0)[cy-128:cy+128, cx-128:cx+128].copy()
                        img_new[:img.shape[0], :img.shape[1]] = img
                else:
                    print(f'path for {path} does not exist, please check')
                imgs.append(img_new)
                
        features = np.array(row[self.features], dtype=np.float32)
        features[np.isnan(features)] = 0
        return np.array(imgs), features ,label
    

    
if __name__ == '__main__':
    train_df = pd.read_csv('./final_data2.csv')
    train_df['bbox_endzone'] = train_df['bbox_endzone'].progress_apply(process_bbox)
    train_df['bbox_sideline'] = train_df['bbox_sideline'].progress_apply(process_bbox)
    print('finished loading data')
    
    train_G1 = train_df.loc[(train_df['contact']==1) & (train_df['G_flug']==True)]
    train_G0 = train_df.loc[(train_df['contact']==0) & (train_df['G_flug']==True)]
    train_P1 = train_df.loc[(train_df['contact']==1) & (train_df['G_flug']==False)]
    train_P0 = train_df.loc[(train_df['contact']==0) & (train_df['G_flug']==False)]
    
    train_G1_set = MyDataset(train_G1[:500].reset_index())
    train_G0_set = MyDataset(train_G0[:1000].reset_index())
    train_P1_set = MyDataset(train_P1[:1500].reset_index())
    train_P0_set = MyDataset(train_P0[:2000].reset_index())

    train_G1_loader = DataLoader(train_G1_set, batch_size=100, num_workers = 8, shuffle=False, pin_memory=False)
    train_G0_loader = DataLoader(train_G0_set, batch_size=100, num_workers = 8, shuffle=False, pin_memory=False)
    train_P1_loader = DataLoader(train_P1_set, batch_size=100, num_workers = 8, shuffle=False, pin_memory=False)
    train_P0_loader = DataLoader(train_P0_set, batch_size=100, num_workers = 8, shuffle=False, pin_memory=False)
    
    start = time.time()
    for i, (batch_G1, batch_G0, batch_P1, batch_P0) in enumerate(zip(cycle(train_G1_loader),cycle(train_G0_loader),cycle(train_P1_loader),train_P0_loader)):
        imgs1, features1, labels1 = batch_G1
        imgs2, features2, labels2 = batch_G0
        imgs3, features3, labels3 = batch_P1
        imgs4, features4, labels4 = batch_P0
        print(i)
    print(f'time taken = {time.time()-start}')