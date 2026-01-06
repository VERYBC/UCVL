# Build same-area test set in XIAN-Visloc based on AMS

import rasterio
import os
import math
import glob
import shutil
import cv2
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.transform import Affine
from pyproj import Transformer
from sklearn.neighbors import BallTree
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd

def latlon_to_radians(coords):
    return np.radians(coords)

def find_nearest_satellite_haversine(uav_coords, sat_coords, k=1):
    tree = BallTree(latlon_to_radians(sat_coords), metric='haversine')
    dist, ind = tree.query(latlon_to_radians(uav_coords), k=k)
    dist_km = dist * 6378.137 * 1000
    return ind, dist_km

def crop_tif_to_tiles(tif_path, out_dir, id, coord_range, tile_size=512, stride=256, tile_id = 1, record_txt_path=None, target_size=None):

    os.makedirs(out_dir, exist_ok=True)

    LT_lat, LT_lon, RB_lat, RB_lon = coord_range
    coordinates = []

    with rasterio.open(tif_path) as src:
        width = src.width
        height = src.height
        transform = src.transform

        tile_id = tile_id

        top_list = list(range(0, height - tile_size + 1, stride))
        left_list = list(range(0, width - tile_size + 1, stride))

        # Check
        if (height - tile_size) % stride != 0:
            top_list.append(height - tile_size)
        if (width - tile_size) % stride != 0:
            left_list.append(width - tile_size)

        total_tiles = len(top_list ) * len(left_list)

        print('The tile num is: {}'.format(total_tiles))
        with tqdm(total=total_tiles, desc="Building the satellite database") as pbar:
            for top in top_list:
                for left in left_list:
                    window = Window(left, top, tile_size, tile_size)

                    tile = src.read(window=window)

                    # Resize to target resolution
                    if target_size is not None:
                        bands, h, w = tile.shape
                        resized_tile = []
                        for b in range(bands):
                            resized = cv2.resize(tile[b], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                            resized_tile.append(resized)
                        tile = np.stack(resized_tile, axis=0)
                        save_h, save_w = target_size

                        scale_x = save_w / tile_size
                        scale_y = save_h / tile_size
                        new_transform = transform * Affine.scale(1 / scale_x, 1 / scale_y)
                    else:
                        save_h, save_w = tile_size, tile_size
                        new_transform = transform

                    # Construct file name
                    new_tile_name = f"{id:02}_{tile_id:05d}.tif"

                    new_out_path = os.path.join(out_dir, new_tile_name)

                    if os.path.exists(new_out_path):
                        print(f"Skip: {new_out_path} already exists")
                    else:
                        profile = src.profile
                        profile.update({
                            'height': save_h,
                            'width': save_w,
                            'transform': new_transform
                        })

                        with rasterio.open(new_out_path, 'w', **profile) as dst:
                            dst.write(tile)

                    # Calculate the center pixel coordinates → latitude and longitude
                    center_y = top + tile_size / 2
                    center_x = left + tile_size / 2

                    x, y = transform * (center_x, center_y)
                    transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(x, y)

                    coordinates.append(f"{new_tile_name},{lat:.7f},{lon:.7f}")

                    tile_id += 1

                    pbar.update(1)

        if record_txt_path is not None:
            with open(record_txt_path, 'a') as f:
                for line in coordinates:
                    f.write(line + '\n')
    return total_tiles

def pos_drone_to_satellite(drone_data, satellite_data, space_resolution):
    drone_coordinates = drone_data.iloc[:, [2, 3]].values
    satellite_coordinates = satellite_data.iloc[:, [1, 2]].values
    pos_num = 50
    rank, distance = find_nearest_satellite_haversine(drone_coordinates, satellite_coordinates, pos_num)
    pos_threhold = min_resolution*space_resolution*0.25*math.sqrt(2)
    pairs = []

    for i, row in drone_data.iterrows():
        drone_name = row['file_name']
        satellite_id = rank[i, :]
        satellite_distance = distance[i, :]
        pos_id = satellite_id[np.where(satellite_distance < pos_threhold)[0]]

        if len(pos_id)==0:
            pos_id = np.array([satellite_id[0]])
        pos_name = list(satellite_data.iloc[pos_id, 0].values)

        if len(pos_name)<pos_num:
            for _ in range(pos_num-len(pos_name)):
                pos_name += ['-1']
        pairs.append([drone_name] + pos_name)

    if pairs_path is not None:
        with open(pairs_path, 'w') as f:
            for line in pairs:
                f.write(','.join(line) + '\n')


if __name__ == "__main__":

    # Setting
    data_root = "G:/Data set/Github/Xian_visloc/Xian_Visloc/Xian"

    DATA_LIST = [16,17,18,19]

    tile_size_set = None

    stride_set = None

    Crop = True

    target_size = (392,392) # Scaling satellite image resolution

    satellite_id = [1,2,3,4]

    # Ground sample distance (GSD) may have errors
    space_resolution = [0.32, 0.32, 0.126, 0.134]

    # Satellite image sampling
    for i in range(1,20):

        if i not in DATA_LIST:
            continue

        # Loading path
        file_dir = os.path.join(data_root, f'drone/{i:02}')

        out_path = os.path.join(file_dir, 'satellite_test')

        satellite_coordinates_path = os.path.join(out_path, f'{i:02}_coordinates.csv')

        drone_info_path = os.path.join(file_dir, f'{i:02}.csv')

        pairs_path = os.path.join(out_path, f'pairs.csv')

        # Read drone image information
        if tile_size_set is None:
            drone_path = os.path.join(file_dir, f'drone/{i:02}_0001.jpg')
            drone_image = Image.open(drone_path)
            width, height = drone_image.size
            max_resolution = max(width, height)
            min_resolution = min(width, height)
            tile_size = round(max_resolution * space_resolution[i-16] / 0.247)

        else:
            tile_size = tile_size_set
            max_resolution = tile_size

        # Read corresponding satellite map information
        stride_total = 0
        W_total = 0
        H_total = 0
        W_set = []
        H_set = []
        for id in range(len(satellite_id)):

            image_path = os.path.join(data_root, f'satellite/satellite{satellite_id[id]:02}.tif')

            with rasterio.open(image_path) as src:
                W = src.width
                H = src.height

            W_set.append(W)
            H_set.append(H)

            W_total += W
            H_total += H

        if stride_set is None:
            T = tile_size
            a = W_total - T
            b = H_total - T

            Threshold = 0.04*(a*b)**(1/3)
            strides = np.arange(100, 1500, 5)
            continuous_deriv = -(2 * a * b) / strides ** 3 - 2 * (a + b) / strides ** 2
            idx = np.argmax(np.abs(continuous_deriv) < Threshold)
            stride = strides[idx]

        else:
            stride = stride_set

        current_tile_id = 1

        for id in range(len(satellite_id)):

            # Cut the satellite map and save latitude and longitude information
            if Crop:

                image_path = os.path.join(data_root, f'satellite/satellite{satellite_id[id]:02}.tif')
                sate_df = pd.read_csv(data_root + "/satellite/satellite_coordinates_range.csv")
                row = sate_df.iloc[satellite_id[id] - 1]
                coord_range = [row['LT_lat_map'], row['LT_lon_map'], row['RB_lat_map'], row['RB_lon_map']]

                print(f'\n==========Crop the {i:02}-{satellite_id[id]:02} satellite map==========')

                tile_id = crop_tif_to_tiles(image_path, out_path, id=i, coord_range=coord_range, tile_size=tile_size, stride=stride, tile_id=current_tile_id, record_txt_path=satellite_coordinates_path, target_size=target_size)
                current_tile_id += tile_id

        # Read drone location information
        drone_coordinates_df = pd.read_csv(drone_info_path)
        satellite_coordinates_df = pd.read_csv(satellite_coordinates_path, header=None)

        # Find the nearest coordinates and record the positive sample
        pos_drone_to_satellite(drone_data=drone_coordinates_df, satellite_data=satellite_coordinates_df, space_resolution=space_resolution[i-16])


