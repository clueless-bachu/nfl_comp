import pandas as pd
import numpy as np
import random
import os
import re

def seed_everything(seed = 42):
    """
    Seeds all available libraries
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
def expand_contact_id(df):
    """
    Splits the contact id in sample submission into separate colums: game, play, player1 and player 2.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

def distance_calc(df_combo, use_cols):
    """
    Takes a df and calculates the distances wherever available
    """
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        x_param_arr = np.full(len(index), np.nan)
        y_param_arr = np.full(len(index), np.nan)
        mag_param_arr = np.full(len(index), np.nan)
        ori_param_arr = np.full(len(index), np.nan)
        
        rel_param_x = np.abs(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
        rel_param_y = np.abs(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        
        tmp_mag_param = np.sqrt(np.square(rel_param_x)+ np.square(rel_param_y))
        tmp_ori_param = np.arctan(rel_param_y/(rel_param_x + 1e-6))*180/np.pi
        
        x_param_arr[index] = rel_param_x
        y_param_arr[index] = rel_param_y
        mag_param_arr[index] = tmp_mag_param
        ori_param_arr[index] = tmp_ori_param
        
        df_combo['rel_pos_x'] = x_param_arr
        df_combo['rel_pos_y'] = y_param_arr
        df_combo['rel_pos_mag'] = mag_param_arr
        df_combo['rel_pos_ori'] = ori_param_arr
    return df_combo

def relative_motion(df_combo, use_cols, param = 'speed'):
    """
    Calculate either relative velocity or acceleration
    """
    if (param in use_cols) & ("direction" in use_cols):
        index = df_combo[param+'_2'].notnull()
        x_param_arr = np.full(len(index), np.nan)
        y_param_arr = np.full(len(index), np.nan)
        mag_param_arr = np.full(len(index), np.nan)
        ori_param_arr = np.full(len(index), np.nan)
        
        x_comp_1 = df_combo.loc[index, param + '_1']*np.sin(df_combo.loc[index, 'direction_1']*np.pi/180)
        y_comp_1 = df_combo.loc[index, param + '_1']*np.cos(df_combo.loc[index, 'direction_1']*np.pi/180)
        
        x_comp_2 = df_combo.loc[index, param + '_2']*np.sin(df_combo.loc[index, 'direction_2']*np.pi/180)
        y_comp_2 = df_combo.loc[index, param + '_2']*np.cos(df_combo.loc[index, 'direction_2']*np.pi/180)
        
        rel_param_x = np.abs(x_comp_1 - x_comp_2)
        rel_param_y = np.abs(y_comp_1 - y_comp_2)
        
        tmp_mag_param = np.sqrt(np.square(rel_param_x)+ np.square(rel_param_y))
        tmp_ori_param = np.arctan(rel_param_y/(rel_param_x + 1e-6))*180/np.pi
        
        x_param_arr[index] = rel_param_x
        y_param_arr[index] = rel_param_y
        mag_param_arr[index] = tmp_mag_param
        ori_param_arr[index] = tmp_ori_param
        
        df_combo['rel_'+param + '_x'] = x_param_arr
        df_combo['rel_'+param + '_y'] = y_param_arr
        df_combo['rel_'+param + '_mag'] = mag_param_arr
        df_combo['rel_'+param + '_ori'] = ori_param_arr
        
        return df_combo

def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    """
    Combines the information in labels data and the tracking data to get tracking data for nfl player 1 and nfl 
    player2 for each STEP (not frame). This includes anything in use_cols
    """
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
   
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    
    # Add distance, speed, aceleration
    df_combo = distance_calc(df_combo, use_cols,)
    df_combo = relative_motion(df_combo, use_cols, 'speed')
    df_combo = relative_motion(df_combo, use_cols, 'acceleration')
        
#     Add is ground tag    
    df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
    
    thresh = 2
    
    df_combo = df_combo.query(f'not rel_pos_mag>{thresh}').reset_index(drop=True)
    df_combo['frame'] = (df_combo['step']/10*59.94+5*59.94).astype('int')+1 # does it matter to go one frame down
    return df_combo

def process_bbox(np_str):
    x = re.sub(' +', ' ', np_str.replace(']','').replace('[','')).split('\n')
    x = [i.split() for i in x]
    for i in range(len(x)):
        for j in range(2):
            try:
                x[i][j] = float(x[i][j])
            except:
                x[i][j] = np.nan
#     x =  pd.DataFrame(x).interpolate(limit_direction='both').values
    return np.array(x)

def save_df(df, path):
    chunks = np.array_split(df.index, 100) # split into 100 chunks

    for chunck, subset in enumerate(tqdm(chunks)):
        if chunck == 0: # first row
            df.loc[subset].to_csv(path, mode='w', index=True)
        else:
            df.loc[subset].to_csv(path, header=None, mode='a', index=True)