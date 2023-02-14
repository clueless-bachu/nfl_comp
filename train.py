from helper import *
import logging
from albumentations.pytorch import ToTensorV2
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
from tqdm import tqdm

CFG = {
    'seed': 42,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'num_workers': 8,  # 0 means do not use multiprocessing
    'batch_size': 32,
    'num_epochs': 10,
    'saver_mode': 'all',
    'es_patience': 6,
    'rop_factor': 0.9,
    'rop_patience': 20000,
    'run_name': 'resnet50_v2',
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


class ValDataset(Dataset):
    def __init__(self, df, aug, one_hot_transform, feature_cols=['rel_pos_x',
                                                                 'rel_pos_y', 'rel_pos_mag', 'rel_pos_ori', 'rel_speed_x', 'rel_speed_y',
                                                                 'rel_speed_mag', 'rel_speed_ori', 'rel_acceleration_x',
                                                                 'rel_acceleration_y', 'rel_acceleration_mag', 'rel_acceleration_ori',
                                                                 'G_flug', 'orientation_1', 'orientation_2']):

        self.df = df
        self.features = feature_cols
        self.aug = aug
        self.one_hot_transform = one_hot_transform

    def __len__(self):
        return len(self.df)

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

        row = self.df.iloc[idx]
        mid_frame = row['frame']

        label = float(row['contact'])
        imgs = []
        for view in ['Endzone', 'Sideline']:
            video = row['game_play'] + f'_{view}.mp4'
            frames = [mid_frame - window +
                      i for i in range(0, 2*window+1, frames_to_skip)]

            bbox_col = 'bbox_endzone' if view == 'Endzone' else 'bbox_sideline'
            bboxes = row[bbox_col][::frames_to_skip].astype(np.int32)

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
                        img = cv2.imread(path, 0)[
                            cy-76:cy+180, cx-128:cx+128].copy()
                        img_new[:img.shape[0], :img.shape[1]] = img
                    else:
                        img = cv2.imread(path, 0)[
                            cy-128:cy+128, cx-128:cx+128].copy()
                        img_new[:img.shape[0], :img.shape[1]] = img
                imgs.append(img_new)

        img = np.array(imgs).transpose(1, 2, 0)
        img = self.aug(image=img)["image"]

        features = np.array(row[self.features], dtype=np.float32)
        features[np.isnan(features)] = 0

        """
        rel_pos_x                0
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
        if row['G_flug']:
            features[6] = row['speed_1']
            features[7] = row['direction_1']
            features[10] = row['acceleration_1']
            features[11] = row['direction_1']

            features[4] = row['speed_1']*np.sin(row['direction_1']*np.pi/180)
            features[5] = row['speed_1']*np.cos(row['direction_1']*np.pi/180)
            features[8] = row['acceleration_1'] * \
                np.sin(row['direction_1']*np.pi/180)
            features[9] = row['acceleration_1'] * \
                np.cos(row['direction_1']*np.pi/180)
        features = self.normalize_features(features)

        team_pos = np.array(
            row[['team_1', 'position_1', 'team_2', 'position_2']].fillna('Ground'))
        team_pos = self.one_hot_transform.transform(
            [team_pos]
        ).toarray()[0]

        return img, torch.from_numpy(np.hstack((features, team_pos)).astype(np.float32)), torch.as_tensor(label)


class TrainDataset(Dataset):
    def __init__(self, df, aug, one_hot_transform, feature_cols=['rel_pos_x',
                                                                 'rel_pos_y', 'rel_pos_mag', 'rel_pos_ori', 'rel_speed_x', 'rel_speed_y',
                                                                 'rel_speed_mag', 'rel_speed_ori', 'rel_acceleration_x',
                                                                 'rel_acceleration_y', 'rel_acceleration_mag', 'rel_acceleration_ori',
                                                                 'G_flug', 'orientation_1', 'orientation_2']):

        self.df = df
        self.features = feature_cols
        self.aug = aug
        self.one_hot_transform = one_hot_transform

    def __len__(self):
        return max([len(df) for df in self.df])*4

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

        df_num = idx % 4
        index = (idx//4) % len(self.df[df_num])

        row = self.df[df_num].iloc[index]
        mid_frame = row['frame']

        label = float(row['contact'])
        imgs = []
        for view in ['Endzone', 'Sideline']:
            video = row['game_play'] + f'_{view}.mp4'
            frames = [mid_frame - window +
                      i for i in range(0, 2*window+1, frames_to_skip)]

            bbox_col = 'bbox_endzone' if view == 'Endzone' else 'bbox_sideline'
            bboxes = row[bbox_col][::frames_to_skip].astype(np.int32)

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
                        img = cv2.imread(path, 0)[
                            cy-76:cy+180, cx-128:cx+128].copy()
                        img_new[:img.shape[0], :img.shape[1]] = img
                    else:
                        img = cv2.imread(path, 0)[
                            cy-128:cy+128, cx-128:cx+128].copy()
                        img_new[:img.shape[0], :img.shape[1]] = img
                imgs.append(img_new)

        img = np.array(imgs).transpose(1, 2, 0)
        img = self.aug(image=img)["image"]

        features = np.array(row[self.features], dtype=np.float32)
        features[np.isnan(features)] = 0

        """
        rel_pos_x                0
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
        if row['G_flug']:
            features[6] = row['speed_1']
            features[7] = row['direction_1']
            features[10] = row['acceleration_1']
            features[11] = row['direction_1']

            features[4] = row['speed_1']*np.sin(row['direction_1']*np.pi/180)
            features[5] = row['speed_1']*np.cos(row['direction_1']*np.pi/180)
            features[8] = row['acceleration_1'] * \
                np.sin(row['direction_1']*np.pi/180)
            features[9] = row['acceleration_1'] * \
                np.cos(row['direction_1']*np.pi/180)
        features = self.normalize_features(features)

        team_pos = np.array(
            row[['team_1', 'position_1', 'team_2', 'position_2']].fillna('Ground'))
        team_pos = self.one_hot_transform.transform(
            [team_pos]
        ).toarray()[0]

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
    def __init__(self, test_df, aug, criterion, transform, verbose=True):
        self.test_set = ValDataset(
            test_df, aug=aug, one_hot_transform=transform)
        self.verbose = verbose
        self.test_loader = DataLoader(
            self.test_set, batch_size=CFG['batch_size'], num_workers=CFG['num_workers'], shuffle=False, pin_memory=True, persistent_workers=bool(CFG['num_workers']))
        self.criterion = criterion

    def validate(self, model, iteration, logger, tb):
        y_hat = []
        y = []
        loss = 0
        model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                imgs, features, labels = batch

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

            stats, val_mathew_corr,  val_acc = get_stats(
                loss, y, y_hat, cur_iter=f"Val@{iteration}", logger=logger)

            tb.add_scalar("Val Loss", loss, iteration)
            tb.add_scalar("Val  Accuracy", val_acc,
                          iteration)
            tb.add_scalar("Val Mathew Correlation",
                          val_mathew_corr, iteration)

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
            model, iteration=iteration, logger=logger, tb=tb)
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
    logger.debug(f"yhat--> {y_hat}")
    logger.debug(f"y--> {y}")

    stats = f'Iteration: {cur_iter} || Loss: {loss:.5f} || mat_corr: {mathew_corr:.5f} || acc: {acc:.5f} || EOL'
    return stats.replace("\n", ""),  mathew_corr, acc


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            'resnet50', pretrained=True, num_classes=250, in_chans=26)
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
        self.fc = nn.Sequential(
            nn.Linear(64+250, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, img, feature):
        img = self.backbone(img)
        feature = self.mlp(feature)
        y = torch.sigmoid(self.fc(torch.cat([img, feature], dim=1)))
        return y.flatten()


def main():
    tb_name_path = CFG['run_name'] + "_" + \
        str(datetime.now()).replace(" ", '-').replace(':', '-').split('.')[0]

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

    np.random.seed(CFG['seed'])
    val_plays = np.random.choice(df['game_play'].unique(), size=2)
    val_set = df.apply(lambda row: row[df['game_play'].isin(val_plays)])
    train_set = df.apply(lambda row: row[~df['game_play'].isin(val_plays)])

    train_G1 = train_set.loc[(df['contact'] == 1) & (
        train_set['G_flug'] == True)].reset_index()
    train_G0 = train_set.loc[(df['contact'] == 0) & (
        train_set['G_flug'] == True)].reset_index()
    train_P1 = train_set.loc[(df['contact'] == 1) & (
        train_set['G_flug'] == False)].reset_index()
    train_P0 = train_set.loc[(df['contact'] == 0) & (
        train_set['G_flug'] == False)].reset_index()

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

    train_set = TrainDataset([train_G1, train_P1, train_P0, train_G0],
                             aug=train_aug, one_hot_transform=one_hot)

    logger.info("Creating dataloader")
    train_loader = DataLoader(
        train_set, batch_size=CFG['batch_size'], num_workers=CFG['num_workers'], shuffle=True, pin_memory=True, persistent_workers=bool(CFG['num_workers']))

    logger.info("Created the dataloader")
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=CFG['rop_factor'], patience=CFG['rop_patience'], verbose=True
    )

    cl_args = {
        "EarlyStopping": {
            'patience': CFG['es_patience']
        },
        "ModelSaver": {
            'save_mode': CFG['saver_mode'],
            'path_name': tb_name_path,
        },
        "Validator": {
            "test_df": [val_set],
            "aug": valid_aug,
            "criterion": criterion,
            "transform": one_hot,
            "verbose": True
        },
    }

    logger.info(f"Callback Arguments:\n{cl_args}")
    cl = Callback(cl_args)

    model = Model()
    model.to('cuda')
    logger.info(f"Model for this run:\n{model}")
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    for cur_epochs in range(CFG['num_epochs']):
        for cur_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            logger.debug("Starting new iteration")
            imgs1, features1, labels1 = batch

            imgs = imgs1.to(0, non_blocking=True)
            feats = features1.to(0, non_blocking=True)
            y = labels1.to(0, non_blocking=True)

            logger.debug(f"imgs shape: {imgs.shape}")
            logger.debug(f"{imgs}")
            logger.debug(f'This is the labels: {labels1}')
            logger.debug(f'This is the features: {feats}')

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                y_hat = model(imgs, feats)
                loss = criterion(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step(loss)

            logger.debug("Updated weights")
            y = y.cpu().detach().numpy()
            y_hat = y_hat.cpu().detach().numpy()
            train_stats, train_mathew_corr, train_acc = get_stats(
                loss, y, y_hat, logger=logger, cur_iter=cur_iter)

            logger.info(f'{train_stats}')

            tb.add_scalar("Train Loss", loss.item(), cur_iter)
            tb.add_scalar("Train Accuracy", train_acc, cur_iter)
            tb.add_scalar("Train Mathew Correlation",
                          train_mathew_corr, cur_iter)

            logger.debug("Logged stats")
            logger.debug("Moving to next iteration")

        if cl.callback(model, cur_epochs, logger=logger, tb=tb):
            break
        # Converting back to training model
        model.train()

    logger.info("Completed number of iterations")
    tb.close()


if __name__ == '__main__':
    main()
