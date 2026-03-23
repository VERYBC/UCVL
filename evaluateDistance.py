import os
import json
import math
import argparse
import scipy.io
import torch
import matplotlib
import pickle
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Evaluate Distance')
parser.add_argument('--root_dir', default='D:/Python_project/Data/Xian_visloc/', type=str, help='./test_data')
parser.add_argument('--model_path', default='./tmp_data', type=str, help='./model_path')
parser.add_argument('--K', nargs='+', default=[1, 3, 5, 10], type=int, help='./test_data')
parser.add_argument('--M', default=5e3, type=str, help='./test_data')
parser.add_argument('--TEST_LIST', nargs='+', default=[2,4,5,15], type=int, help='Test List')
parser.add_argument('--test_mode', default='', type=str, help='if 1v1')
parser.add_argument('--if_print', action='store_true', help='if output')
parser.add_argument('--if_center', action='store_true', help='if center')
parser.add_argument('--read_index', action='store_true', help='if read index')
parser.add_argument('--mode', default="1", type=str, help='1:drone->satellite 2:satellite->drone')
opts = parser.parse_args()

opts.drone_dir = []
opts.satellite_dir = []

for i in opts.TEST_LIST:
    opts.drone_dir.append(os.path.join(opts.root_dir, f'{i:02}/{i:02}.csv'))
    if 'UAV_visloc' in opts.root_dir:
        opts.satellite_dir.append(os.path.join(opts.root_dir, f'{i:02}/satellite_test'+ opts.test_mode + f'/{i:02}_coordinates.csv'))
    else:
        opts.satellite_dir.append(os.path.join(opts.root_dir, f'{i:02}/satellite_test'+ opts.test_mode + f'/{i:02}_coordinates.csv'))

configDict_drone = {}
configDict_satellite = {}
for i in range(len(opts.TEST_LIST)):
    drone_coordinates_df = pd.read_csv(opts.drone_dir[i])
    drone_names = drone_coordinates_df.iloc[:, 1].values
    if 'UAV_visloc' in opts.root_dir:
        drone_coordinates = drone_coordinates_df.iloc[:, [3, 4]].values
    else:
        drone_coordinates = drone_coordinates_df.iloc[:, [2, 3]].values

    for k, v in zip(drone_names, drone_coordinates):
        configDict_drone[k] = v

    satellite_coordinates_df = pd.read_csv(opts.satellite_dir[i], header=None)
    satellite_names = satellite_coordinates_df.iloc[:, 0].values
    satellite_coordinates = satellite_coordinates_df.iloc[:, [1, 2]].values
    for k, v in zip(satellite_names, satellite_coordinates):
        configDict_satellite[k] = v

#####################################################################
# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated

######################################################################
if opts.mode == "1":
    result = scipy.io.loadmat(opts.model_path + '/features.mat')
else:
    result = scipy.io.loadmat(opts.model_path + '/features_.mat')

query_feature = torch.FloatTensor(result['img_features_query'])
query_label = result['ids_query']
query_paths = result['paths_query'][0]

gallery_feature = torch.FloatTensor(result['img_features_gallery'])
gallery_label = result['ids_gallery']
gallery_paths = result['paths_gallery'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images and return topK index
def sort_img(qf, ql, gf, gl, K):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = []

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index[:K]

def getLatitudeAndLongitude(imgPath, query = False):

    if opts.mode == "1": # query-drone
        if query:
            configDict = configDict_drone
        else:
            configDict = configDict_satellite
    else: # query-satellite
        if query:
            configDict = configDict_satellite
        else:
            configDict = configDict_drone

    if isinstance(imgPath, list):
        posInfo = [configDict[os.path.basename(p)] for p in imgPath]
    else:
        posInfo = configDict[os.path.basename(imgPath)]

    return posInfo

def euclideanDistance(query, gallery):
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)
    A = gallery - query
    A_T = A.transpose()
    distance = np.matmul(A, A_T)
    # d = geodesic((query[0], query[1]), (gallery[0][0], gallery[0][1])).meters
    mask = np.eye(distance.shape[0], dtype=np.bool_)
    distance = distance[mask]
    distance = np.sqrt(distance.reshape(-1))
    return distance

def evaluateSingle(distance, K):
    # maxDistance = max(distance) + 1e-14
    # weight = np.ones(K) - np.log(range(1, K + 1, 1)) / np.log(opts.M * K)
    weight = np.ones(K) - np.array(range(0, K, 1))/K
    # m1 = distance / maxDistance
    m2 = np.exp(-distance*opts.M)
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result

def latlog2meter(lata, loga, latb, logb):
    # lat latitude  log longitude

    EARTH_RADIUS = 6378.137
    PI = math.pi
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    
    return distance

def evaluate_SDM(indexOfTopK, queryIndex, K):
    query_path = query_paths[queryIndex][0]
    galleryTopKPath = [gallery_paths[i][0] for i in indexOfTopK[:K]]
    # get position information including latitude and longitude
    queryPosInfo = getLatitudeAndLongitude(query_path, query = True)
    galleryTopKPosInfo = getLatitudeAndLongitude(galleryTopKPath)
    # compute Euclidean distance of query and gallery
    distance = euclideanDistance(queryPosInfo, galleryTopKPosInfo)
    # compute single query evaluate result
    P = evaluateSingle(distance, K)
    return P

def evaluate_MA(indexOfTopK, queryIndex):
    query_path = query_paths[queryIndex][0]
    minimum_distance_meter = 99999
    total_distance_meter = 0
    for id in indexOfTopK:
        galleryTopKPath = gallery_paths[id][0]
        # get position information including latitude and longitude
        queryPosInfo = getLatitudeAndLongitude(query_path, query = True)
        galleryTopKPosInfo = getLatitudeAndLongitude(galleryTopKPath)
        # get real distance
        distance_meter = latlog2meter(queryPosInfo[0],queryPosInfo[1],galleryTopKPosInfo[0],galleryTopKPosInfo[1])
        total_distance_meter +=  distance_meter
        if distance_meter < minimum_distance_meter:
            minimum_distance_meter = distance_meter
            minimum_PosInfo = [galleryTopKPosInfo[1], galleryTopKPosInfo[0]]
            index = id + 1
    mean_distance_meter = total_distance_meter/len(indexOfTopK)
    return minimum_distance_meter, minimum_PosInfo, index

indexOfTopK_list = []
index_path = opts.model_path+'/indexOfTopK_list.pkl'

if os.path.exists(index_path) and opts.read_index:
    with open(index_path, 'rb') as f:
        indexOfTopK_list = pickle.load(f)
    print("The index file exists and has been read")
else:
    for i in range(len(query_label)):
        indexOfTopK = sort_img(query_feature[i], query_label[i], gallery_feature, gallery_label, 100)
        indexOfTopK_list.append(indexOfTopK)

SDM_dict = {}
TOP1_Pos = []
TOP5_Pos = []
TOP10_Pos = []
for K in tqdm(range(1, 11, 1)):
    metric = 0
    for i in range(len(query_label)):
        P_ = evaluate_SDM(indexOfTopK_list[i], i, K)
        metric += P_
    metric = metric / len(query_label)
    SDM_dict[K] = metric

MA_dict = {}
for meter in tqdm(range(0,1000,5)):
    MA_K = 0
    Meter_mean_1 = 0
    Meter_mean_5 = 0
    for i in range(len(query_label)):
        MA_meter_1, top1_pos, top1_id = evaluate_MA(indexOfTopK_list[i][:1],i)
        MA_meter_5, top5_pos, top5_id = evaluate_MA(indexOfTopK_list[i][:5], i)
        MA_meter_10, top10_pos,top10_id  = evaluate_MA(indexOfTopK_list[i][:10], i)
        if MA_meter_1<meter:
            MA_K+=1
        Meter_mean_1 += MA_meter_1
        Meter_mean_5 += MA_meter_5

        if meter == 0:
            TOP1_Pos.append([query_label[i][0]]+top1_pos+[top1_id])
            TOP5_Pos.append([query_label[i][0]]+top5_pos+[top5_id])
            TOP10_Pos.append([query_label[i][0]]+top10_pos+[top10_id])
                            
    MA_K = MA_K/len(query_label)
    MA_dict[meter]=MA_K

Meter_mean_1 = Meter_mean_1/len(query_label)
MA_dict['TOP1_Distance'] = Meter_mean_1

Meter_mean_5 = Meter_mean_5/len(query_label)
MA_dict['TOP5_Distance'] = Meter_mean_5

if opts.if_print:
    print("\nSDM@{} = {:.2f}%".format(1, SDM_dict[1] * 100))
    print("\nTOP1_Distance = {:.2f} m".format(Meter_mean_1))
    print("\nTOP5_Distance = {:.2f} m \n".format(Meter_mean_5))

if not os.path.exists(opts.model_path):
    os.makedirs(opts.model_path)

with open(opts.model_path+"/SDM@K(1,10).json", 'w') as F:
    json.dump(SDM_dict, F, indent=4)

with open(opts.model_path+"/MA@K_200.json", 'w') as F:
    json.dump(MA_dict, F, indent=4)

# Save search position information
top1_path = opts.model_path + f"/TOP1_Pos_{opts.TEST_LIST[0]}.csv"
top5_path = opts.model_path + f"/TOP5_Pos_{opts.TEST_LIST[0]}.csv"

if os.path.exists(top1_path):
    os.remove(top1_path)

if os.path.exists(top5_path):
    os.remove(top5_path)


with open(top1_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['label', 'longitude', 'latitude', 'id'])
    writer.writerows(TOP1_Pos)

with open(top5_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['label', 'longitude', 'latitude', 'id'])
    writer.writerows(TOP5_Pos)
