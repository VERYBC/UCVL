import cv2
import os
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_match.LightGlue.lightglue import LightGlue, SuperPoint
from feature_match.LightGlue.lightglue .utils import load_image, rbd
from feature_match.LoFTR.src.loftr import LoFTR, default_cfg
from feature_match.Roma.romatch import roma_outdoor
from feature_match.Romav2.romav2 import RoMaV2

os.environ["TORCHDYNAMO_DISABLE"]  = "1"

def process_resize(w, h, resize=[-1]):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]
    return w_new, h_new

def frame2tensor(frame, device, resize=[-1]):
    w, h = frame.shape[1], frame.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    frame = cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def haversine(loga, lata, logb, latb):

    EARTH_RADIUS = 6378.137
    PI = math.pi

    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance

if __name__ == '__main__':
    DATA_LIST = [3]
    resolution = [1959,1671,2223]
    size = 1024  # loftr image size
    space_resolution = [0.126,0.215,0.143]
    model = 'romav2' # 'lightglue' | 'loftr' | 'roma' | 'romav2' |
    satellite_type = 'TOP1'
    if_location = True
    data_root =  "./data/Real-world filght data"
    save_path = 'feature_match/results/' + satellite_type + f'_location_{DATA_LIST[0]:02}_' + model + '.csv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_keypoints = -1  # -1 keep all keypoints
    keypoint_threshold = 0.01  # Remove keypoints with low confidence. Set to -1 to keep all keypoints.
    nms_radius = 4  # Non-maxima suppression: keypoints with similar responses in a small neighborhood are removed.
    sinkhorn_iterations = 20  # Number of Sinkhorn iterations for matching.
    match_threshold = 0.55 # Remove matches with low confidence. Set to -1 to keep all matches.
    superglue = 'outdoor'  # The SuperGlue model to use. Either 'indoor' or 'outdoor'.

    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

    if if_location:

        # load model
        if 'lightglue' in model:
                extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
                matcher = LightGlue(features='superpoint').eval().to(device)
        elif model == 'loftr':
            loftr = LoFTR(config=default_cfg)
            loftr.load_state_dict(torch.load("feature_match/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
            loftr = loftr.eval().cuda()
        elif model == 'roma':
            roma_model = roma_outdoor(device=device)
        elif model == 'romav2':
            romav2_model = RoMaV2()

        # Read the image path and coarse matching results
        query_folder = data_root + f'/{DATA_LIST[0]:02}/drone_392'
        query_paths = [os.path.join(query_folder, f)
                       for f in sorted(os.listdir(query_folder))
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

        satellite_folder = data_root + f'/{DATA_LIST[0]:02}/satellite_test'
        satellite_paths = [os.path.join(satellite_folder, f) for f in os.listdir(satellite_folder)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

        pos_info_path = './tmp_data/'+satellite_type+f'_Pos_{DATA_LIST[0]}.csv'
        pos_coordinates_df = pd.read_csv(pos_info_path)

        results = []
        mkpts_nums = []
        delta_dists = []
        success_num = 0
        with torch.no_grad():
            start = 0
            for i, query_path in enumerate(tqdm(query_paths[start:], desc="localization："), start=start):

                satellite_index = pos_coordinates_df.iloc[i, 3]
                satellite_pos = pos_coordinates_df.iloc[i, 1:3].values

                if i==start:
                    dist = 0
                    delta_dist = 0
                    optimal_location = satellite_pos
                    previous_location = satellite_pos
                    previous_dx_m = 0
                    previous_dy_m = 0

                # satellite_image = load_image(satellite_paths[satellite_index - 1]).to(device)
                satellite_path = data_root + f'/{DATA_LIST[0]:02}/satellite_test/{DATA_LIST[0]:02}_{satellite_index:05d}.tif'

                # read images
                query_image = load_image(query_path).to(device)
                satellite_image = load_image(satellite_path).to(device)

                image0 = query_image
                image1 = satellite_image
                H_A, W_A = image0.shape[-2:]
                H_B, W_B = image1.shape[-2:]

                if model == 'loftr':
                    image0 = frame2tensor(cv2.imread(query_path, cv2.IMREAD_GRAYSCALE), device, [size])
                    image1 = frame2tensor(cv2.imread(satellite_path, cv2.IMREAD_GRAYSCALE), device, [size])

                    H_A, W_A = image0.shape[-2:]
                    H_B, W_B = image1.shape[-2:]

                elif model == 'roma' or model == 'romav2':
                    image0 = cv2.imread(query_path)
                    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)

                    image1 = cv2.imread(satellite_path)
                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

                    H_A, W_A = image0.shape[:2]
                    H_B, W_B = image1.shape[:2]

                # Lightglue
                if 'lightglue' in model:
                    feats0 = extractor.extract(query_image)
                    feats1 = extractor.extract(satellite_image)
                    matches01 = matcher({'image0': feats0, 'image1': feats1})
                    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
                    matches = matches01['matches']  # indices with shape (K,2)
                    mkpts0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
                    mkpts1 = feats1['keypoints'][matches[..., 1]].cpu().numpy() # coordinates in image #1, shape (K,2)

                # LoFTR
                if model == 'loftr':
                    batch = {'image0': image0, 'image1': image1}
                    loftr(batch)
                    mkpts0 = batch['mkpts0_f'].cpu().numpy()
                    mkpts1 = batch['mkpts1_f'].cpu().numpy()

                # Roma
                if model == 'roma':
                    warp, certainty = roma_model.match(query_path, satellite_path, device=device)
                    matches, certainty = roma_model.sample(warp, certainty)
                    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
                    mkpts0 = kptsA.cpu().numpy()
                    mkpts1 = kptsB.cpu().numpy()

                # Romav2
                if model == 'romav2':
                    preds = romav2_model.match(query_path, satellite_path)
                    matches, overlaps, precision_AB, precision_BA = romav2_model.sample(preds, 10000)

                    kptsA, kptsB = romav2_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
                    mkpts0 = kptsA.cpu().numpy()
                    mkpts1 = kptsB.cpu().numpy()

                mkpts_nums.append(len(mkpts0))

                if i == start:
                    print(H_A, W_A, H_B, W_B)

                if len(mkpts0) >= 4:
                    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)

                    if H is not None:

                        GSD = 0.247 * resolution[DATA_LIST[0]-1] / H_B

                        center_query = np.array([[W_A / 2, H_A / 2]], dtype=np.float32).reshape(-1, 1, 2)
                        center_satellite_pixel = cv2.perspectiveTransform(center_query, H)

                        dx_px = center_satellite_pixel[0, 0, 0] - (H_B / 2)
                        dy_px = center_satellite_pixel[0, 0, 1] - (W_B / 2)

                        dx_m = dx_px * GSD
                        dy_m = dy_px * GSD

                        lon0, lat0 = satellite_pos
                        lon = lon0 + dx_m / (111320 * np.cos(np.deg2rad(lat0)))
                        lat = lat0 - dy_m / 110540

                        current_location = [lon, lat]

                        optimal_dist  = haversine(current_location[0], current_location[1], optimal_location[0], optimal_location[1])
                        previous_dist = haversine(current_location[0], current_location[1], previous_location[0], previous_location[1])

                        if optimal_dist<50 or previous_dist<20:
                            results.append([os.path.basename(query_path), current_location[0], current_location[1]])
                            optimal_location = current_location
                            if 204>i>=122:
                                success_num += 1
                        else:
                            satellite_dist = haversine(satellite_pos[0], satellite_pos[1], optimal_location[0], optimal_location[1])
                            if satellite_dist>600:
                                results.append([os.path.basename(query_path), optimal_location[0], optimal_location[1]])
                            else:
                                results.append([os.path.basename(query_path), satellite_pos[0], satellite_pos[1]])
                                optimal_location = satellite_pos

                        previous_location = current_location

                    else:
                        results.append([os.path.basename(query_path),  satellite_pos[0],  satellite_pos[1]])

        df = pd.DataFrame(results, columns=['Image', 'Longitude', 'Latitude'])
        df.to_csv(save_path, index=False)
        print('Save the results to '+ save_path)

    print(f"Number of successfully located frames: {success_num}/84")
    print(f"successful positioning rate          : {success_num/84 * 100:.2f}%")


