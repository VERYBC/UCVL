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
    top1_dfs  = []
    top5_dfs  = []

    for i in range(1, 4):
        if i not in DATA_LIST:
            continue
        file_dir = os.path.join(data_root, f'{i:02}')
        drone_info_path = os.path.join(file_dir, f'{i:02}.csv')
        drone_coordinates_df = pd.read_csv(drone_info_path)
        drone_dfs.append(drone_coordinates_df)
        drone_ids.append(i)

        # Read top1 positioning information
        top1_info_path = f'tmp_data/TOP1_Pos_{i}.csv'
        top1_coordinates_df = pd.read_csv(top1_info_path)
        top1_dfs.append(top1_coordinates_df)

        # Read top5 positioning information
        top5_info_path = f'tmp_data/TOP5_Pos_{i}.csv'
        top5_coordinates_df = pd.read_csv(top5_info_path)
        top5_dfs.append(top5_coordinates_df)


    for i, (df, top1_df, top5_df, drone_id) in enumerate(zip(drone_dfs, top1_dfs, top5_dfs, drone_ids)):
        true_lat = df.iloc[:, 2].to_numpy()
        true_lon = df.iloc[:, 3].to_numpy()

        top1_lat = top1_df.iloc[:, 2].to_numpy()
        top1_lon = top1_df.iloc[:, 1].to_numpy()

        top5_lat = top5_df.iloc[:, 2].to_numpy()
        top5_lon = top5_df.iloc[:, 1].to_numpy()

        # ---- Top1 ----
        success_errors1 = []
        status1 = classify_points(top1_lat, top1_lon, dist_thres=500)
        errors1 = np.array([haversine(true_lat[j], true_lon[j], top1_lat[j], top1_lon[j]) for j in range(len(true_lat))])

        smoothed_errors1 = errors1.copy()
        last_valid_error = errors1[0]
        success_errors1.append(errors1[0])
        for j in range(1, len(errors1)):
            if status1[j] == 'L':
                smoothed_errors1[j] = last_valid_error
            else:
                last_valid_error = errors1[j]
                success_errors1.append(errors1[j])
        success_errors1 = np.array(success_errors1)

        # ---- Top5 ----
        success_errors5 = []
        status5 = classify_points(top5_lat, top5_lon, dist_thres=500)
        errors5 = np.array([haversine(true_lat[j], true_lon[j], top5_lat[j], top5_lon[j]) for j in range(len(true_lat))])

        smoothed_errors5 = errors5.copy()
        last_valid_error = errors5[0]
        success_errors5.append(errors5[0])
        for j in range(1, len(errors5)):
            if status5[j] == 'L':
                smoothed_errors5[j] = last_valid_error
            else:
                last_valid_error = errors5[j]
                success_errors5.append(errors5[j])
        success_errors5 = np.array(success_errors5)


    # Output positioning results
    # top1
    print(f'top1_mean_error   : {success_errors1.mean():.2f} m')
    print(f'top1_lost_fraction: {(1-len(success_errors1)/len(errors1)) * 100:.2f}%')
    # top5
    print(f'top5_mean_error   : {success_errors5.mean():.2f} m')
    print(f'top5_lost_fraction: {(1-len(success_errors5)/len(errors5)) * 100:.2f}%')

