import os
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def classify_points(lat_list, lon_list, dist_thres=200):
    status = ['H']
    last_high_lat, last_high_lon = lat_list[0], lon_list[0]

    for i in range(1, len(lat_list)):
        d_high = haversine(lat_list[i], lon_list[i], last_high_lat, last_high_lon)

        if d_high > dist_thres:
            status.append('L')
        else:
            status.append('H')
            last_high_lat, last_high_lon = lat_list[i], lon_list[i]

    return status

if __name__ == '__main__':

    DATA_LIST = [3]
    data_root = "./data/Real-world filght data"

    drone_dfs = []
    drone_ids = []
    top5_dfs  = []
    roma_dfs  = []
    romav2_dfs  = []
    loftr_dfs  = []
    lightglue_dfs = []
    optimal_distances = []

    for i in range(1, 4):
        if i not in DATA_LIST:
            continue
        file_dir = os.path.join(data_root, f'{i:02}')
        drone_info_path = os.path.join(file_dir, f'{i:02}.csv')
        drone_coordinates_df = pd.read_csv(drone_info_path)
        drone_dfs.append(drone_coordinates_df)
        drone_ids.append(i)

        # top5
        top5_info_path = f'./tmp_data/TOP5_Pos_{i}.csv'
        top5_coordinates_df = pd.read_csv(top5_info_path)
        top5_dfs.append(top5_coordinates_df)

        # roma
        roma_info_path = f'./feature_match/results/TOP1_location_{i:02}_roma.csv'
        roma_coordinates_df = pd.read_csv(roma_info_path)
        roma_dfs.append(roma_coordinates_df)

        # loftr
        loftr_info_path = f'./feature_match/results/TOP1_location_{i:02}_loftr.csv'
        loftr_coordinates_df = pd.read_csv(loftr_info_path)
        loftr_dfs.append(loftr_coordinates_df)

        # lightglue
        lightglue_info_path = f'./feature_match/results/TOP1_location_{i:02}_lightglue.csv'
        lightglue_coordinates_df = pd.read_csv(lightglue_info_path)
        lightglue_dfs.append(lightglue_coordinates_df)

        # romav2
        romav2_info_path = f'./feature_match/results/TOP1_location_{i:02}_romav2.csv'
        romav2_coordinates_df = pd.read_csv(romav2_info_path)
        romav2_dfs.append(romav2_coordinates_df)


    for i, (df, top5_df, romav2_df, roma_df, loftr_df, lightglue_df, drone_id) in enumerate(zip(drone_dfs, top5_dfs, romav2_dfs, roma_dfs, loftr_dfs, lightglue_dfs, drone_ids)):

        lat = df.iloc[:, 2].values
        lon = df.iloc[:, 3].values

        top5_lon = top5_df.iloc[:, 1].values
        top5_lat = top5_df.iloc[:, 2].values

        romav2_lon = romav2_df.iloc[:, 1].values
        romav2_lat = romav2_df.iloc[:, 2].values

        roma_lon = roma_df.iloc[:, 1].values
        roma_lat = roma_df.iloc[:, 2].values

        loftr_lon = loftr_df.iloc[:, 1].values
        loftr_lat = loftr_df.iloc[:, 2].values

        lightglue_lon = lightglue_df.iloc[:, 1].values
        lightglue_lat = lightglue_df.iloc[:, 2].values

    start = 122
    end = len(roma_lon)-17

    romav2_errors = haversine(lat[start:end], lon[start:end], romav2_lat[start:end],romav2_lon[start:end])
    romav2_mean_error = np.mean(romav2_errors)
    print(f'romav2_mean_error: {romav2_mean_error}')

    roma_errors = haversine(lat[start:end], lon[start:end], roma_lat[start:end],roma_lon[start:end])
    roma_mean_error = np.mean(roma_errors)
    print(f'roma_mean_error: {roma_mean_error}')

    lightglue_errors = haversine(lat[start:end], lon[start:end], lightglue_lat[start:end],lightglue_lon[start:end])
    lightglue_mean_error = np.mean(lightglue_errors)
    print(f'lightglue_mean_error: {lightglue_mean_error}')

    loftr_errors = haversine(lat[start:end], lon[start:end], loftr_lat[start:end],loftr_lon[start:end])
    loftr_mean_error = np.mean(loftr_errors)
    print(f'loftr_mean_error: {loftr_mean_error}')
