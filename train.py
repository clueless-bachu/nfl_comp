from helper import *
import logging
from albumentations.pytorch import ToTensorV2
from multiprocessing import Pool, set_start_method
import albumentations as A
import timm
from datetime import datetime
from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import matthews_corrcoef, accuracy_score
import os
import numpy as np
import pandas as pd
import random
import cv2
from math import ceil
from tqdm import tqdm


CFG = {
    'seed': 42,
    'test_size': 1000,
    'lr': 1e-3,
    'num_workers': 8,  # 0 means do not use multiprocessing
    'batch_size': 64,
    'iterations': 25000*10,
    'val_wait': 200*10,
    'saver_mode': 'all',
    'es_patience': 100,
    'rop_factor': 0.9,
    'rop_patience': 700,
    'run_name': 'baseline_3',
    'log_level': logging.INFO,
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


categorical_data_for_fitting = [
    ['home', 'CB', 'home', 'CB', ],
    ['home', 'DE', 'home', 'DE', ],
    ['home', 'FS', 'home', 'FS', ],
    ['home', 'TE', 'home', 'TE', ],
    ['home', 'ILB', 'home', 'ILB', ],
    ['home', 'OLB', 'home', 'OLB', ],
    ['home', 'T', 'home', 'T', ],
    ['home', 'G', 'home', 'G', ],
    ['home', 'C', 'home', 'C', ],
    ['home', 'QB', 'home', 'QB', ],
    ['home', 'WR', 'home', 'WR', ],
    ['home', 'RB', 'home', 'RB', ],
    ['home', 'NT', 'home', 'NT', ],
    ['home', 'DT', 'home', 'DT', ],
    ['home', 'MLB', 'home', 'MLB', ],
    ['home', 'SS', 'home', 'SS', ],
    ['home', 'OT', 'home', 'OT', ],
    ['home', 'LB', 'home', 'LB', ],
    ['home', 'OG', 'home', 'OG', ],
    ['home', 'SAF', 'home', 'SAF', ],
    ['home', 'DB', 'home', 'DB', ],
    ['home', 'LS', 'home', 'LS', ],
    ['home', 'K', 'home', 'K', ],
    ['home', 'P', 'home', 'P', ],
    ['home', 'FB', 'home', 'FB', ],
    ['home', 'S', 'home', 'S', ],
    ['home', 'DL', 'Ground', 'DL', ],
    ['away', 'HB', 'away', 'HB', ],
    ['away', 'HB', 'away', 'Ground', ],
]


def read_image(path, cx, cy, view, aug):
    img_new = np.zeros((1, 256, 256), dtype=np.float32)
    if os.path.isfile(path):
        if view == 'Endzone':
            img = cv2.imread(path, 0)[
                cy-76:cy+180, cx-128:cx+128].copy()
            img_new[0, :img.shape[0], :img.shape[1]] = img
        else:
            img = cv2.imread(path, 0)[
                cy-128:cy+128, cx-128:cx+128].copy()
            img_new[0, :img.shape[0], :img.shape[1]] = img

    return aug(image=img_new.transpose(1, 2, 0))['image'][0]


class MyDataset(Dataset):
    def __init__(self, df1, df2, df3, df4, logger, aug, one_hot_transform, train=True, feature_cols=['rel_pos_x',
                                                                                                     'rel_pos_y', 'rel_pos_mag', 'rel_pos_ori', 'rel_speed_x', 'rel_speed_y',
                                                                                                     'rel_speed_mag', 'rel_speed_ori', 'rel_acceleration_x',
                                                                                                     'rel_acceleration_y', 'rel_acceleration_mag', 'rel_acceleration_ori',
                                                                                                     'G_flug', 'orientation_1', 'orientation_2']):

        self.df1 = df1
        self.df2 = df2
        self.df3 = df3
        self.df4 = df4
        self.logger = logger
        self.features = feature_cols
        self.aug = aug
        self.train = train
        self.one_hot_transform = one_hot_transform

    def __len__(self):
        return max(len(self.df1), len(self.df2), len(self.df3), len(self.df4))//(CFG['batch_size']//4)

    def get_rows(self, lnum, unum, df_num):
        df_map = {
            1: self.df1,
            2: self.df2,
            3: self.df3,
            4: self.df4,
        }
        self.logger.debug(f"df{df_num} with lnum:{lnum}, unum:{unum}")
        lnum = lnum % len(df_map[df_num])
        unum = unum % len(df_map[df_num])
        self.logger.debug(f"df{df_num} with lnum:{lnum}, unum:{unum}")

        if lnum < unum:
            self.logger.debug(f"return is {df_map[df_num][lnum:unum].shape}")
            return df_map[df_num][lnum:unum]
        else:
            self.logger.debug(
                f"return is {pd.concat([df_map[df_num][lnum:], df_map[df_num][:unum]]).shape}")
            if self.train:
                return pd.concat([df_map[df_num][lnum:], df_map[df_num][:unum]])
            else:
                return df_map[df_num][lnum:]

    def normalize_features(self, features):
        """
        normalizes the features of the players

       'rel_pos_x',
       'rel_pos_y', 'rel_pos_mag', 'rel_pos_ori', 'rel_speed_x', 'rel_speed_y',
       'rel_speed_mag', 'rel_speed_ori', 'rel_acceleration_x',
       'rel_acceleration_y', 'rel_acceleration_mag', 'rel_acceleration_ori',
       'G_flug', 'orientation_1', 'orientation_2'
        """
        features /= 100
        features[3] /= 3.6
        features[7] /= 3.6
        features[11] /= 3.6
        features[13] /= 3.6
        features[14] /= 3.6
        return features

    def __getitem__(self, idx):
        window = 24
        frames_to_skip = 4

        self.logger.debug(f"idx that is causing issue: {idx}")

        row1 = self.get_rows(
            idx*(CFG['batch_size']//4), (idx+1)*(CFG['batch_size']//4), 1).reset_index(drop=True)
        row2 = self.get_rows(
            idx*(CFG['batch_size']//4), (idx+1)*(CFG['batch_size']//4), 2).reset_index(drop=True)
        row3 = self.get_rows(
            idx*(CFG['batch_size']//4), (idx+1)*(CFG['batch_size']//4), 3).reset_index(drop=True)
        row4 = self.get_rows(
            idx*(CFG['batch_size']//4), (idx+1)*(CFG['batch_size']//4), 4).reset_index(drop=True)

        row = pd.concat([row1, row2, row3, row4]).reset_index(drop=True)
        self.logger.debug(f"row colums: {row.columns}")
        self.logger.debug(f"Row shape:{row.shape}")
        mid_frame = row['frame']
        self.logger.debug(f"mid frames shape:{len(mid_frame)}")
        label = np.array(row['contact']).astype(np.float32)
        self.logger.debug(f"label:{len(label)}")
        args = []
        for i in range(len(row)):
            for view in ['Endzone', 'Sideline']:
                video = row.iloc[i]['game_play'] + f'_{view}.mp4'
                cur_mid_frame = mid_frame.iloc[i]
                frames = [cur_mid_frame - window +
                          next_frame for next_frame in range(0, 2*window+1, frames_to_skip)]
                bbox_col = 'bbox_endzone' if view == 'Endzone' else 'bbox_sideline'
                self.logger.debug(
                    f"bbox details:\n{row.iloc[i][bbox_col][::frames_to_skip]}")
                bboxes = row.iloc[i][bbox_col][::frames_to_skip].astype(
                    np.int32)

                if bboxes.sum() <= 0:
                    args += [('dummy', 0, 0, view, self.aug)]*len(frames)
                    continue

                for frame_iter, frame in enumerate(frames):
                    cx, cy = bboxes[frame_iter]
                    path = f'./work/train_frames/{video}_{frame:04d}.jpg'
                    args.append((path, cx, cy, view, self.aug))

        self.logger.debug(f"sizeof args:{len(args)}")
        with Pool(CFG['num_workers']) as pool:
            imgs = list(pool.starmap(read_image, args))
            pool.close()
        
        img = torch.stack(imgs).reshape(len(row), 26, 256, 256)
        self.logger.debug(f"processed imgs:{img.shape}")
        features = np.array(row[self.features], dtype=np.float32)
        features[np.isnan(features)] = 0

        """
        rel_pos_x                0fork
        rel_pos_y                1
        rel_pos_mag              2
        rel_pos_ori              3
        rel_speed_x              4
        rel_speed_y              5
        rel_speed_mag            6
        rel_speed_ori            7
        rel_acceleration_x       8
        rel_acceleration_y       9
        rel_acceleration_mag     10
        rel_acceleration_ori     11 
        """
        for i in range(len(row)):
            if row.iloc[i]['G_flug']:
                features[i, 6] = row.iloc[i]['speed_1']
                features[i, 7] = row.iloc[i]['direction_1']
                features[i, 10] = row.iloc[i]['acceleration_1']
                features[i, 11] = row.iloc[i]['direction_1']

                features[i, 4] = row.iloc[i]['speed_1'] * \
                    np.sin(row.iloc[i]['direction_1']*np.pi/180)
                features[i, 5] = row.iloc[i]['speed_1'] * \
                    np.cos(row.iloc[i]['direction_1']*np.pi/180)
                features[i, 8] = row.iloc[i]['acceleration_1'] * \
                    np.sin(row.iloc[i]['direction_1']*np.pi/180)
                features[i, 9] = row.iloc[i]['acceleration_1'] * \
                    np.cos(row.iloc[i]['direction_1']*np.pi/180)

            features[i, :] = self.normalize_features(features[i])
        self.logger.debug(f"processed features:{features.shape}")

        team_pos = np.array(
            row[['team_1', 'position_1', 'team_2', 'position_2']].fillna('Ground'))
        team_pos = self.one_hot_transform.transform(team_pos).toarray()
        # gc.collect()
        return img, torch.from_numpy(np.hstack((features, team_pos)).astype(np.float32)), torch.as_tensor(label)


class EarlyStopping():
    """
    A class which decides to stop early or not based on patience and Mathew Correaltion Metrci

    Use this on validation set only
    """

    def __init__(self, patience):
        self.patience = patience
        self.best_metric = -np.inf
        self.counter = 0

    def stop(self, metric_val, logger):

        flag = False
        if metric_val > self.best_metric:
            self.best_metric = metric_val
            self.counter = 0
            flag = True
        else:
            self.counter += 1
        logger.info(
            f"Inside Early Stopping, counter/patience = {self.counter}/{self.patience}")

        if self.counter > self.patience:
            return True, flag
        else:
            return False, flag


class ModelSaver():

    """
    save mode: can be one of {'none', 'best', 'indexed', 'all'}
    """

    def __init__(self, save_mode, path_name):
        self.save_mode = save_mode

        self.path = './model_checkpoints/' + path_name
        if self.save_mode is not None:
            os.mkdir(self.path)

    def save(self, model, index, logger, best: bool):
        if self.save_mode == 'best':
            if best:
                logger.info(
                    f"Saving Best Model at {self.path+ '/best_model.pth'}")
                torch.save(model, self.path + '/best_model.pth')
                return
            else:
                return

        elif self.save_mode == 'indexed':
            cur_save_path = self.path + '/iteration' + \
                str(index).zfill(7) + '.pth'
            logger.info(f"Saving lateset model at {cur_save_path}")
            torch.save(model, cur_save_path)
            return

        elif self.save_mode == 'all':
            cur_save_path = self.path + '/iteration' + \
                str(index).zfill(7) + '.pth'
            logger.info(f"Saving lateset model at {cur_save_path}")
            torch.save(model, cur_save_path)
            if best:
                logger.info(
                    f"Saving Best Model at {self.path+ '/best_model.pth'}")
                torch.save(model, self.path + '/best_model.pth')
                return
            else:
                return


class Validator():
    def __init__(self, df1, df2, df3, df4, aug, logger, criterion, transform, verbose=True):
        self.test_set = MyDataset(
            df1, df2, df3, df4, logger=logger, aug=aug, train=False, one_hot_transform=transform)
        self.verbose = verbose
        self.counter = 0
        self.criterion = criterion

    def validate(self, model, train_iteration, logger, tb):
        y_hat = []
        y = []
        loss = 0
        logger.debug("converting testloader to iter")
        logger.debug("done converting testloader to iter")
        model.eval()
        with torch.no_grad():
            for val_iteration in range(ceil(CFG['test_size']/CFG['batch_size'])):

                imgs, features, labels = self.test_set[val_iteration]
                imgs = imgs.to(0, non_blocking=True)
                features = features.to(0, non_blocking=True)
                labels = labels.to(0, non_blocking=True)

                preds = model(imgs, features)

                loss += self.criterion(preds,
                                       labels).cpu().detach().numpy().ravel()[0]
                y.append(labels.cpu().detach().numpy())
                y_hat.append(preds.cpu().detach().numpy())

            logger.debug(f"Combined val labels:\n{y}")
            logger.debug(f"Combined val preds:\n{y_hat}")
            y = np.hstack(y)
            y_hat = np.hstack(y_hat)
            logger.debug(f"Combined val labels:\n{y}")
            logger.debug(f"Combined val preds:\n{y_hat}")

            loss = loss/CFG['batch_size']

            stats, val_mathew_corr,  val_acc, _, _ = get_stats(
                loss, y, y_hat, cur_iter=f"Val@{train_iteration}", logger=logger)

            tb.add_scalar("Val Loss", loss, train_iteration)
            tb.add_scalar("Val  Accuracy", val_acc,
                          train_iteration)
            tb.add_scalar("Val Mathew Correlation",
                          val_mathew_corr, train_iteration)
            if self.verbose:
                logger.info(f"{stats}")

        return val_mathew_corr


class Callback():
    def __init__(self, args):
        self.valer = Validator(**args['Validator'])
        self.es = EarlyStopping(**args['EarlyStopping'])
        self.ms = ModelSaver(**args['ModelSaver'])

    def callback(self, model, iteration, logger, tb):
        logger.info("Validating Data")
        metric = self.valer.validate(
            model, train_iteration=iteration, logger=logger, tb=tb)
        stop, best = self.es.stop(metric, logger=logger)
        if best:
            logger.info("New Best Model !!")
        if stop:
            logger.info("Early Stopping Triggered !!")
        logger.info(f'stop:{stop} isBest: {best}')
        self.ms.save(model, iteration, logger, best)
        logger.info(f"completed validation")
        return stop


def get_stats(loss, y, y_pred, logger, cur_iter='val', thresh=0.5):
    """
    Gets the stats for a particular batch
    """
    y_hat = (y_pred > thresh)*1.0
    mathew_corr = matthews_corrcoef(y, y_hat)
    acc = accuracy_score(y, y_hat)

    size = len(y_hat)
    classwise_mathew_corr = []
    classwise_acc = []
    for i in range(4):
        y_cl = y[i*(size//4):(i+1)*(size//4)]
        y_hat_cl = y_hat[i*(size//4):(i+1)*(size//4)]
        logger.debug(f"{i}")
        logger.debug(f"this is y_cl\n{y_cl}")
        logger.debug(f"this is y_cl\n{y_hat_cl}")
        logger.debug(f"cl mtcorr:{matthews_corrcoef(y_cl, y_hat_cl)}")
        logger.debug(f"cl acc:{accuracy_score(y_cl, y_hat_cl)}")
        classwise_mathew_corr.append(matthews_corrcoef(y_cl, y_hat_cl))
        classwise_acc.append(accuracy_score(y_cl, y_hat_cl))

    stats = f'Iteration: {cur_iter} || Loss: {loss:.5f} || mat_corr: {mathew_corr:.5f} || acc: {acc:.5f}'
    stats += f""" 
|| G1_mat_corr: {classwise_mathew_corr[0]:.5f} || G0_mat_corr: {classwise_mathew_corr[1]:.5f} || P1_mat_corr: {classwise_mathew_corr[2]:.5f} || P0_mat_corr: {classwise_mathew_corr[3]:.5f}"""
    stats += f"""
||G1_acc: {classwise_acc[0]:.5f} || G0_acc: {classwise_acc[1]:.5f} || P1_acc: {classwise_acc[2]:.5f} || P0_acc: {classwise_acc[3]:.5f} EOL
    """

    return stats.replace("\n", ""),  mathew_corr, acc, classwise_mathew_corr, classwise_acc


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            'resnet50', pretrained=False, num_classes=250, in_chans=26)
        self.mlp = nn.Sequential(
            nn.Linear(77, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(64+250, 1)

    def forward(self, img, feature):
        img = self.backbone(img)
        feature = self.mlp(feature)
        y = torch.sigmoid(self.fc(torch.cat([img, feature], dim=1)))
        return y.flatten()


def main():
    tb_name_path = CFG['run_name'] + "_" + \
        str(datetime.now()).replace(" ", '-').replace(':', '-').split('.')[0]
    set_start_method('fork')

    logger = logging.getLogger(__name__)
    logger.setLevel(CFG['log_level'])

    file_handler = logging.FileHandler(f'./logs/{tb_name_path}.log')
    file_handler.setFormatter(logging.Formatter(
        "[%(name)s::%(asctime)s :: %(levelname)s][%(filename)s:%(lineno)d] %(message)s"))

    logger.addHandler(file_handler)

    tb = SummaryWriter(
        f"/home/vasista/Desktop/nfl-player-contact-detection/tensorboard/{tb_name_path}")
    logger.info(f"Unique name of this run is: {tb_name_path}")
    logger.info(f'Here are the configs for this run:\n{CFG}')

    seed_everything(CFG['seed'])
    logger.info("Everything is Seeded!")

    one_hot = OneHotEncoder()
    one_hot.fit(categorical_data_for_fitting)
    logger.info(f"this is our one_hot encoder: {one_hot.categories_}")

    logger.info("Starting the train program, loading Data")
    df = pd.read_csv('./final_data2.csv')
    df['bbox_endzone'] = df['bbox_endzone'].apply(process_bbox)
    df['bbox_sideline'] = df['bbox_sideline'].apply(process_bbox)
    logger.info("Loaded Train dataset")

    df_G1 = df.loc[(df['contact'] == 1) & (df['G_flug'] == True)]
    df_G0 = df.loc[(df['contact'] == 0) & (df['G_flug'] == True)]
    df_P1 = df.loc[(df['contact'] == 1) & (df['G_flug'] == False)]
    df_P0 = df.loc[(df['contact'] == 0) & (df['G_flug'] == False)]

    random_state = 42

    train_G1, test_G1 = train_test_split(
        df_G1, test_size=CFG['test_size']//4, random_state=random_state)
    train_G0, test_G0 = train_test_split(
        df_G0, test_size=CFG['test_size']//4, random_state=random_state)
    train_P1, test_P1 = train_test_split(
        df_P1, test_size=CFG['test_size']//4, random_state=random_state)
    train_P0, test_P0 = train_test_split(
        df_P0, test_size=CFG['test_size']//4, random_state=random_state)

    logger.info("Split the dataset into train and val sets")

    train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.Normalize(mean=[0.], std=[1.]),
        ToTensorV2()
    ])

    valid_aug = A.Compose([
        A.Normalize(mean=[0.], std=[1.]),
        ToTensorV2()
    ])

    train_set = MyDataset(train_G1.reset_index(drop=True), train_G0.reset_index(drop=True), train_P1.reset_index(drop=True), train_P0.reset_index(drop=True), logger=logger,
                          aug=train_aug, one_hot_transform=one_hot)

    logger.info("Creating dataloader")

    logger.info("Created the dataloader")
    cl_args = {
        "EarlyStopping": {
            'patience': CFG['es_patience']
        },
        "ModelSaver": {
            'save_mode': CFG['saver_mode'],
            'path_name': tb_name_path,
        },
        "Validator": {
            "df1": test_G1,
            "df2": test_G0,
            "df3": test_P1,
            "df4": test_P0,
            "logger": logger,
            "aug": valid_aug,
            "criterion": nn.BCELoss(),
            "transform": one_hot,
            "verbose": True
        },
    }

    logger.info(f"Callback Arguments:\n{cl_args}")
    cl = Callback(cl_args)

    # model = Model()
    model = torch.load("./model_checkpoints/baseline_3_2023-02-06-08-19-29/best_model.pth")
    model.to('cuda')
    logger.info(f"Model for this run:\n{model}")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=CFG['rop_factor'], patience=CFG['rop_patience'], verbose=True
    )

    num_iters = CFG['iterations']
    validate_wait = CFG['val_wait']
    model.train()

    imgs, feats, labels = train_set[7348]
    for cur_iter in tqdm(range(num_iters+1)):
        logger.debug("Starting new iteration")
        imgs, features, labels = train_set[cur_iter]

        imgs = imgs.to(0, non_blocking=True)
        y = labels.to(0, non_blocking=True)
        feats = features.to(0, non_blocking=True)

        logger.debug(f"imgs shape: {imgs.shape}")
        logger.debug(f"features shape: {features}")
        logger.debug(f"labels: {labels}")

        optimizer.zero_grad()

        y_hat = model(imgs, feats)
        loss = criterion(y_hat, y)

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        logger.debug("Updated weights")
        y = y.cpu().detach().numpy()
        y_hat = y_hat.cpu().detach().numpy()
        train_stats, train_mathew_corr, train_acc, train_classwise_mathew_corr, train_classwise_acc = get_stats(
            loss, y, y_hat, logger=logger, cur_iter=cur_iter)

        # del imgs, feats, y
        logger.info(f'{train_stats}')

        tb.add_scalar("Train Loss", loss.item(), cur_iter)
        tb.add_scalar("Train Accuracy", train_acc, cur_iter)
        tb.add_scalar("Train Mathew Correlation",
                      train_mathew_corr, cur_iter)

        tb.add_scalar("Train Mathew Correlation G1",
                      train_classwise_mathew_corr[0], cur_iter)
        tb.add_scalar("Train Mathew Correlation P1",
                      train_classwise_mathew_corr[1], cur_iter)
        tb.add_scalar("Train Mathew Correlation P0",
                      train_classwise_mathew_corr[2], cur_iter)
        tb.add_scalar("Train Mathew Correlation G0",
                      train_classwise_mathew_corr[3], cur_iter)

        tb.add_scalar("Train Accuracy G1",
                      train_classwise_acc[0], cur_iter)
        tb.add_scalar("Train Accuracy P1",
                      train_classwise_acc[1], cur_iter)
        tb.add_scalar("Train Accuracy P0",
                      train_classwise_acc[2], cur_iter)
        tb.add_scalar("Train Accuracy G0",
                      train_classwise_acc[3], cur_iter)

        logger.debug("Logged stats")

        if cur_iter % validate_wait == 0:
            if cl.callback(model, cur_iter, logger=logger, tb=tb):
                break
            # Converting back to training model
            model.train()
        logger.debug("Moving to next iteration")

    logger.info("Completed number of iterations")
    tb.close()


if __name__ == '__main__':
    main()
