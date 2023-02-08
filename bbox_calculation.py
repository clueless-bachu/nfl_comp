from multiprocessing import  Pool
from functools import partial
import concurrent.futures

import os
import sys
import glob
import numpy as np
import pandas as pd
import random
import math
import gc
import cv2
from tqdm import tqdm
tqdm.pandas()
import time
from helper import *

combined_df = pd.read_csv('./combined_df.csv')
helmet_data= pd.read_csv('./train_baseline_helmets.csv')


def add_bboxes_details(frame, player1, player2, game_play, view):
    
    bboxes = []
    window = 24
    query = f"frame >= {frame-window}  and frame <= {frame+window} and game_play == '{game_play}'"
    if player2 == 'G':
        query += f"  and nfl_player_id in ({player1}, 'G')"
    else:
        query += f"  and nfl_player_id in ({player1}, {player2})"
    filt = helmet_data.query(query)

    tmp = filt[filt["view"] == view]
    tmp = tmp.groupby('frame')[['left','width','top','height']].mean()
    tmp['centre_x'] = tmp['left'] + tmp['width']/2
    tmp['centre_y'] = tmp['top'] + tmp['height']/2
    frame_list = tmp.index # these are the available frames
    for fr in range(frame - window, frame + window + 1):
        if fr in frame_list:
            x, y = tmp.loc[fr][['centre_x','centre_y']]
            bboxes.append([x, y])
        else:
            bboxes.append([np.nan, np.nan])
    return np.array(bboxes)
   


if __name__ == '__main__':
    
#     combined_df = combined_df.loc[: 100]
    
    combined_df['bbox_endzone'] = combined_df.progress_apply(
    lambda x: add_bboxes_details(x['frame'],x['nfl_player_id_1'],x['nfl_player_id_2'],x['game_play'], 'Endzone'),
    axis=1
       )
    combined_df['bbox_sideline'] = combined_df.progress_apply(
    lambda x: add_bboxes_details(x['frame'],x['nfl_player_id_1'],x['nfl_player_id_2'],x['game_play'], 'Sideline'),
    axis=1
       )
    combined_df.to_csv('./final_data.csv')
    