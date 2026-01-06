# Build UAV-Visloc training set based on AMS

import rasterio
import os
import cv2
import math
import glob
from multiprocessing import Pool
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
    dist_km = dist * 6378.137 * 1000  # 地球半径转换为公里
    return ind, dist_km

def merge_tifs(tif_paths, out_merged_path):
    src_files_to_mosaic = [rasterio.open(p) for p in tif_paths]
    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(out_merged_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

def process_tile(args):
    src_path, tile_size, left, top, transform, LT_lat, LT_lon, lat_per_pixel, lon_per_pixel, out_dir, id, tile_id, target_size = args
    
    with rasterio.open(src_path) as src:
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

        profile = src.profile
        profile.update({
            'height': save_h,
            'width': save_w,
            'transform': new_transform
        })

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
    
    return f"{new_tile_name},{lat:.7f},{lon:.7f}"

def crop_tif_to_tiles_parallel(tif_path, out_dir, id, coord_range, tile_size=512, stride=256, record_txt_path=None, num_workers=8, target_size=None):

    os.makedirs(out_dir, exist_ok=True)

    LT_lat, LT_lon, RB_lat, RB_lon = coord_range

    with rasterio.open(tif_path) as src:
        width = src.width
        height = src.height
        transform = src.transform

        lat_per_pixel = (LT_lat - RB_lat) / height
        lon_per_pixel = (RB_lon - LT_lon) / width

        top_list = list(range(0, height - tile_size + 1, stride))
        left_list = list(range(0, width - tile_size + 1, stride))

        if (height - tile_size) % stride != 0:
            top_list.append(height - tile_size)
        if (width - tile_size) % stride != 0:
            left_list.append(width - tile_size)

        tasks = []
        tile_id = 1
        for top in top_list:
            for left in left_list:

                window = Window(left, top, tile_size, tile_size)
                tasks.append((tif_path, tile_size, left, top, transform,
                              LT_lat, LT_lon, lat_per_pixel, lon_per_pixel,
                              out_dir, id, tile_id, target_size))
                tile_id += 1
    print(f"\n In parallel cropping, there are a total of {len(tasks)} tiles, using {num_workers} processes")

    with Pool(processes=num_workers) as pool:
        coordinates = list(tqdm(pool.imap(process_tile, tasks), total=len(tasks)))

    if record_txt_path:
        with open(record_txt_path, 'w') as f:
            for line in coordinates:
                f.write(line + '\n')

def pos_drone_to_satellite(drone_data, satellite_data, space_resolution):
    drone_coordinates = drone_data.iloc[:, [3, 4]].values
    satellite_coordinates = satellite_data.iloc[:, [1, 2]].values
    pos_num = 10
    rank, distance = find_nearest_satellite_haversine(drone_coordinates, satellite_coordinates, pos_num)
    pos_threhold = min_resolution*space_resolution*0.25*math.sqrt(2)
    pairs = []
    for i, row in drone_data.iterrows():
        drone_name = row['filename']
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

    # setting
    data_root = "D:/Python_project/Data/UAV_visloc"

    # same-area [1,2,3,4,5,6,8,9,]
    # cross-area [3,4,5,6,8,9]
    DATA_LIST = [1,2,3,4,5,6,8,9]

    tile_size_set = None

    stride_set = None

    Crop = True

    # The 9th satellite map needs to be spliced first
    Merge = False

    target_size = (392,392) # Scaling satellite image resolution

    # Ground sample distance (GSD) may have errors
    space_resolution = [0.1,0.1,0.11,0.14,0.125,0.08,0.055,0.14,0.14,0.1,0.19]

    # Satellite image sampling
    for i in range(1,10):

        if i not in DATA_LIST:
            continue

        # Loading path
        file_dir = os.path.join(data_root, f'{i:02}')

        image_path = os.path.join(file_dir, f'satellite{i:02}.tif')

        if i == 9 and Merge:
            tif_list = sorted(glob.glob(file_dir+'/*.tif'))
            merge_tifs(tif_list, file_dir + f'/satellite{i:02}.tif')

        out_path = os.path.join(file_dir, 'satellite')

        satellite_coordinates_path = os.path.join(out_path, f'{i:02}_coordinates.csv')

        drone_info_path = os.path.join(file_dir, f'{i:02}.csv')

        pairs_path = os.path.join(out_path, f'pairs.csv')

        # Read satellite latitude and longitude information
        sate_df = pd.read_csv(data_root + "/satellite_coordinates_range.csv")
        row = sate_df.iloc[i-1]
        coord_range = [row['LT_lat_map'], row['LT_lon_map'], row['RB_lat_map'], row['RB_lon_map']]

        # Read drone image information
        if tile_size_set is None:
            drone_path = os.path.join(file_dir, f'drone/{i:02}_0001.JPG')
            drone_image = Image.open(drone_path)
            width, height = drone_image.size
            max_resolution = max(width, height)
            min_resolution = min(width, height)
            tile_size = round(max_resolution * space_resolution[i-1] / 0.3)
        else:
            tile_size = tile_size_set
            max_resolution  = tile_size

        with rasterio.open(image_path) as src:
            W = src.width
            H = src.height

        if stride_set is None:

            T = tile_size
            a = W - T
            b = H - T

            Threshold = 0.04*(a*b)**(1/3)
            strides = np.arange(100, 1500, 5)
            continuous_deriv = -(2 * a * b) / strides ** 3 - 2 * (a + b) / strides ** 2
            idx = np.argmax(np.abs(continuous_deriv) < Threshold)
            stride = strides[idx]
        else:
            stride = stride_set

        # Cut the satellite map and save latitude and longitude information
        if Crop:
            print(f'\n ==========Crop the {i:02}-th satellite map==========')
            print(f'\n Tile size is : {tile_size}')
            print(f'\n Stride is : {stride}')

            crop_tif_to_tiles_parallel(image_path, out_path, id=i, coord_range=coord_range, tile_size=tile_size, stride=stride, record_txt_path=satellite_coordinates_path, num_workers=8, target_size =target_size)

        # Read drone location information
        satellite_coordinates_df = pd.read_csv(satellite_coordinates_path,header=None)
        drone_coordinates_df = pd.read_csv(drone_info_path)

        # Find the nearest coordinates and record the positive sample
        pos_drone_to_satellite(drone_data=drone_coordinates_df, satellite_data=satellite_coordinates_df, space_resolution=space_resolution[i-1])


