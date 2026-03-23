import os
import cv2
import time
import copy
import random

import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from random import choice
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

def get_data(path_list):
    data = {}
    for path in path_list:
        for root, dirs, files in os.walk(path, topdown=False):
            files = sorted(files) 
            for name in files:
                if 'tif' in name:
                    id = name.replace('_', '').replace('.tif', '')
                    if id not in data:
                        data[id] = {}
                    if '1_1' in path:
                        data[id]["path_1_1"] = os.path.join(root, name)
                    else:
                        data[id]["path"] = os.path.join(root, name)
                elif 'JPG' in name:
                    id = name.replace('_', '').replace('.JPG', '')
                    data[id] = {"path": os.path.join(root, name)}
                elif 'jpg' in name:
                    id = name.replace('_', '').replace('.jpg', '')
                    data[id] = {"path": os.path.join(root, name)}
                elif 'png' in name:
                    id = name.replace('_', '').replace('.png', '')
                    data[id] = {"path": os.path.join(root, name)}
    return data


class VislocDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 transforms=None,
                 sample_ids=None,
                 satellite_train_set=None,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        self.ids = list(self.data_dict.keys())

        self.transforms = transforms
        
        self.given_sample_ids = sample_ids
        
        self.images = []
        self.sample_ids = []
        self.paths = []

        self.gallery_n = gallery_n

        if satellite_train_set is not None:

            match_info_list = []
            match_info_list_1_1 = []

            # Read the matching information between drone and satellite images
            for path in data_folder:
                if_random_crop = '1_1' not in path
                if if_random_crop:
                    match_info_pd = pd.read_csv(path + '/pairs.csv', header=None)
                    match_info_list += list(match_info_pd.iloc[:].values)
                else:
                    match_info_pd = pd.read_csv(path + '/pairs1_1.csv', header=None)
                    match_info_list_1_1 += list(match_info_pd.iloc[:].values)

            match_info_df = pd.DataFrame(match_info_list)
            match_info_df_1_1 = pd.DataFrame(match_info_list_1_1)

            self.pairs = satellite_train_set.pairs

            for pair in self.pairs:

                satellite_id, satellite_path, _, _,_ = pair
                self.images.append(satellite_path)

                if '1_1' in satellite_path:

                    satellite_id_str = satellite_path[-12:]
                    mask = (match_info_df_1_1.iloc[:, 1:] == satellite_id_str).any(axis=1)
                    drone_pos = match_info_df_1_1.loc[mask].iloc[:, 0].tolist()

                    if len(drone_pos) > 0:
                        if 'JPG' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.JPG', '') for x in drone_pos]
                        elif 'jpg' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.jpg', '') for x in drone_pos]
                        elif 'png' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.png', '') for x in drone_pos]
                        if len(drone_pos) < 100:
                            for _ in range(100 - len(drone_pos)):
                                drone_pos.append(0)
                        self.sample_ids.append(drone_pos)

                else:

                    satellite_id_str = satellite_path[-12:]
                    mask = (match_info_df.iloc[:, 1:] == satellite_id_str).any(axis=1)
                    drone_pos = match_info_df.loc[mask].iloc[:, 0].tolist()

                    if len(drone_pos) > 0:
                        if 'JPG' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.JPG', '') for x in drone_pos]
                        elif 'jpg' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.jpg', '') for x in drone_pos]
                        elif 'png' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.png', '') for x in drone_pos]
                        if len(drone_pos) < 100:
                            for _ in range(100 - len(drone_pos)):
                                drone_pos.append(0)
                        self.sample_ids.append(drone_pos)
                    else:
                        self.sample_ids.append([0 for _ in range(100)])

        elif 'satellite' in data_folder[0]:

            match_info_list = []
            match_info_list_1_1 = []

            seq = os.path.basename(os.path.dirname(data_folder[0]))
      
            coordinates_path = os.path.join(data_folder[0], f'{seq}_coordinates.csv')
            coordinates_info_pd = pd.read_csv(coordinates_path, header=None)
            self.coordinates = coordinates_info_pd.iloc[:,1:].values

            # Read the matching information between drone and satellite images
            for path in data_folder:
                if_random_crop = '1_1' not in path
                if if_random_crop:
                    match_info_pd = pd.read_csv(path + '/pairs.csv', header=None)
                    match_info_list += list(match_info_pd.iloc[:].values)
                else:
                    match_info_pd = pd.read_csv(path + '/pairs1_1.csv', header=None)
                    match_info_list_1_1 += list(match_info_pd.iloc[:].values)

            match_info_df = pd.DataFrame(match_info_list)
            match_info_df_1_1 = pd.DataFrame(match_info_list_1_1)

            # Read all satellite images and retrieve corresponding drone images
            for satellite_id in self.data_dict:
                if 'path' in self.data_dict[satellite_id]:
                    satellite_path = self.data_dict[satellite_id]["path"]

                    self.images.append(satellite_path)
                    satellite_id_str = satellite_path[-12:]
                    mask = (match_info_df.iloc[:, 1:] == satellite_id_str).any(axis=1)
                    drone_pos = match_info_df.loc[mask].iloc[:, 0].tolist()

                    if len(drone_pos) > 0:
                        if 'JPG' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.JPG', '') for x in drone_pos ]
                        elif 'jpg' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.jpg', '') for x in drone_pos]
                        elif 'png' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.png', '') for x in drone_pos]
                        if len(drone_pos) < 200:
                            for _ in range(200-len(drone_pos)):
                                drone_pos.append(0)
                        self.sample_ids.append(drone_pos)
                    else:
                        self.sample_ids.append([0 for _ in range(200)])

                if 'path_1_1' in self.data_dict[satellite_id]:
                    satellite_path_1_1 = self.data_dict[satellite_id]["path_1_1"]

                    self.images.append(satellite_path_1_1)
                    satellite_id_str = satellite_path_1_1[-12:]
                    mask = (match_info_df_1_1.iloc[:, 1:] == satellite_id_str).any(axis=1)
                    drone_pos = match_info_df_1_1.loc[mask].iloc[:, 0].tolist()

                    if len(drone_pos) > 0:
                        if 'JPG' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.JPG', '') for x in drone_pos]
                        elif 'jpg' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.jpg', '') for x in drone_pos]
                        elif 'png' in drone_pos[0]:
                            drone_pos = [x.replace('_', '').replace('.png', '') for x in drone_pos]
                        if len(drone_pos) < 100:
                            for _ in range(100-len(drone_pos)):
                                drone_pos.append(0)
                        self.sample_ids.append(drone_pos)

                    else:
                        self.sample_ids.append([0 for _ in range(100)])

        else:
            seq = os.path.basename(os.path.dirname(data_folder[0]))
        
            if 'UAV_visloc' in data_folder[0]:
                coordinates_path = os.path.join(os.path.dirname(data_folder[0]), f'{seq}.csv')
                coordinates_info_pd = pd.read_csv(coordinates_path)
                self.coordinates = coordinates_info_pd.iloc[:, 3:].values
            elif 'Xian_visloc' in data_folder[0]:
                coordinates_path = os.path.join(os.path.dirname(data_folder[0]), f'{seq}.csv')
                coordinates_info_pd = pd.read_csv(coordinates_path)
                self.coordinates = coordinates_info_pd.iloc[:, 2:].values
            elif 'AdM_UAV' in data_folder[0]:
                coordinates_path = os.path.join(os.path.dirname(data_folder[0]), f'{seq}.txt')
                coordinates_info_pd = pd.read_csv(coordinates_path, sep=r"\s+")
                self.coordinates = coordinates_info_pd.iloc[:, [0, 1]].values
                
            for i, sample_id in enumerate(self.ids):
                self.images.append(self.data_dict[sample_id]["path"])
                self.sample_ids.append([sample_id])

    def __getitem__(self, index):
        
        img_path = self.images[index]
        sample_id = self.sample_ids[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = np.array([int(x) for x in sample_id])
        
        return img, label, img_path

    def __len__(self):
        return len(self.images)
    
    def get_sample_ids(self):
        return set(self.sample_ids)
    
    
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    

    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
                                
                             

    train_sat_transforms = A.Compose([A.ImageCompression(quality_range=(90, 100), p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(num_holes_range=(5, 25),hole_height_range=(8, 32),hole_width_range=(8, 32),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])
    
    train_drone_transforms = A.Compose([A.ImageCompression(quality_range=(90, 100), p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
                                        A.OneOf([
                                                 A.AdvancedBlur(p=1.0),
                                                 A.Sharpen(p=1.0),
                                              ], p=0.3),
                                        A.OneOf([
                                                 A.GridDropout(ratio=0.4, p=1.0),
                                                 A.CoarseDropout(num_holes_range=(5, 25), hole_height_range=(int(0.1*img_size[0]), int(0.2*img_size[0])),hole_width_range=(int(0.1*img_size[0]), int(0.2*img_size[0])), p=1.0),
                                              ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])
    
    return val_transforms, train_sat_transforms, train_drone_transforms