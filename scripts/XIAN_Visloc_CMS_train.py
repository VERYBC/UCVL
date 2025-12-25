# Build XIAN-Visloc training set based on CMS

import rasterio
import os
import cv2
import math
import glob
import shutil
from rasterio.merge import merge
from rasterio.windows import Window
from sklearn.neighbors import BallTree
from rasterio.warp import transform as crs_transform
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

def crop_tif_to_tiles(tif_path, out_dir, id, uav_coords, tile_size=512, record_txt_path=None, target_size=None):

    os.makedirs(out_dir, exist_ok=True)

    coordinates = []

    with rasterio.open(tif_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

        tile_id = 1

        lats, lons = zip(*uav_coords)
        xs, ys = crs_transform("EPSG:4326", crs, lons, lats)

        print('\n The tile num is: {}'.format(len(uav_coords)))
        with tqdm(total=len(uav_coords), desc="Building the 1v1 satellite database") as pbar:
            for i in range(len(xs)):
                x, y = xs[i], ys[i]

                px, py = ~transform * (x, y)
                center_x = int(px)
                center_y = int(py)

                top = center_y - tile_size // 2
                left = center_x - tile_size // 2

                if top < 0 or left < 0 or top + tile_size > height or left + tile_size > width:
                    print(f"({lats[i]},{lons[i]}) - out of bounds")
                    top = max(0, min(top, height - tile_size))
                    left = max(0, min(left, width - tile_size))

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
                else:
                    save_h, save_w = tile_size, tile_size

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
                        'transform': transform
                    })

                    with rasterio.open(new_out_path, 'w', **profile) as dst:
                        dst.write(tile)

                coordinates.append(f"{new_tile_name},{lats[i]:.7f},{lons[i]:.7f}")

                tile_id += 1

                pbar.update(1)

        if record_txt_path is not None:
            with open(record_txt_path, 'w') as f:
                for line in coordinates:
                    f.write(line + '\n')

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

        if len(pos_id) == 0:
            pos_id = np.array([satellite_id[0]])
        pos_name = list(satellite_data.iloc[pos_id, 0].values)
        if len(pos_name) < pos_num:
            for _ in range(pos_num - len(pos_name)):
                pos_name += ['-1']
        pairs.append([drone_name] + pos_name)

    if pairs_path is not None:
        with open(pairs_path, 'w') as f:
            for line in pairs:
                f.write(','.join(line) + '\n')


if __name__ == "__main__":

    # Setting
    data_root ="G:/Data set/Github/Xian_visloc/Xian_Visloc/Xian"

    DATA_LIST = list(range(1,16))

    tile_size_set = None

    Crop = True

    target_size = (39,39) # Scaling satellite image resolution

    satellite_id = [1,2,3,1,4,3,4,3,3,4,4,1,1,2,2]

    # Ground sample distance (GSD) may have errors
    space_resolution = [0.32 for _ in range(7)] + [0.107,0.126,0.126,0.143,0.134,0.126,0.116,0.134]

    # Satellite image sampling
    for i in range(1,20):

        if i not in DATA_LIST:
            continue

        # Loading path
        file_dir = os.path.join(data_root, f'drone/{i:02}')

        image_path = os.path.join(data_root, f'satellite/satellite{satellite_id[i-1]:02}.tif')

        out_path = os.path.join(file_dir, 'satellite1_1')

        satellite_coordinates_path = os.path.join(out_path, f'{i:02}_coordinates.csv')

        drone_info_path = os.path.join(file_dir, f'{i:02}.csv')

        pairs_path = os.path.join(out_path, f'pairs1_1.csv')

        # Read satellite latitude and longitude information
        sate_df = pd.read_csv(data_root + "/satellite/satellite_coordinates_range.csv")
        row = sate_df.iloc[satellite_id[i-1]-1]
        coord_range = [row['LT_lat_map'], row['LT_lon_map'], row['RB_lat_map'], row['RB_lon_map']]

        # Read drone image information
        drone_coordinates_df = pd.read_csv(drone_info_path)
        drone_coordinates = drone_coordinates_df.iloc[:, [2, 3]].values

        if tile_size_set is None:
            drone_path = os.path.join(file_dir, f'drone/{i:02}_0001.jpg')
            drone_image = Image.open(drone_path)
            width, height = drone_image.size
            max_resolution = max(width, height)
            min_resolution = min(width, height)
            tile_size = round(max_resolution * space_resolution[i-1] / 0.247)
        else:
            tile_size = tile_size_set

        # Cut the satellite map and save latitude and longitude information
        if Crop:

            print(f'\n ==========Crop the {i:02}-th satellite map==========')
            print(f'\n Tile size is : {tile_size}')

            crop_tif_to_tiles(image_path, out_path, id=i, uav_coords=list(drone_coordinates), tile_size=tile_size, record_txt_path=satellite_coordinates_path, target_size=target_size)

        satellite_coordinates_df = pd.read_csv(satellite_coordinates_path, header=None)

        # 查找最近的坐标，并记录正样本
        pos_drone_to_satellite(drone_data=drone_coordinates_df, satellite_data=satellite_coordinates_df, space_resolution=space_resolution[i-1])





