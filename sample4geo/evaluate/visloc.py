import os
import gc
import torch
import pickle
import copy
import time

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.cluster import KMeans
from ..trainer import predict
from scipy.io import savemat, loadmat
from sklearn.metrics.pairwise import haversine_distances

def match_top_windows(c_qf, c_gf, region_size=6):

    c_qf = torch.as_tensor(c_qf)
    c_gf = torch.as_tensor(c_gf)

    C, H, W = c_gf.shape

    kernel = torch.ones((1, 1, region_size, region_size), device= c_gf.device) / (region_size * region_size)

    feat = c_gf.unsqueeze(0)
    mean_map = F.conv2d(feat, kernel.expand(C, -1, -1, -1), groups=C)
    mean_map = mean_map.squeeze(0)  # (C, h, w)
    norm = torch.sqrt(torch.sum(mean_map ** 2, dim=0, keepdim=True) + 1e-6)
    mean_map = mean_map / norm

    c_qf = c_qf.unsqueeze(-1).unsqueeze(-1)
    sim_map = torch.sum(mean_map * c_qf, dim=0)  # (h, w)

    best_sim, idx = torch.max(sim_map.view(-1), dim=0)
    y, x = divmod(idx.item(), sim_map.shape[1])

    best_coor = (x + region_size // 2, y + region_size // 2)

    distance = (best_coor[1] - (H / 2 + 0.5)) ** 2 + (best_coor[0] - (W / 2 + 0.5)) ** 2

    return best_coor, distance, best_sim.item()

def evaluate(config,
                  model,
                  query_loader,
                  gallery_loader,
                  ranks=[1, 5, 10],
                  save_path='./tmp_data',
                  save_feature=False,
                  read_feature=False,
                  if_print=True,
                  if_center=False,
                  if_aggregation=False,
                  cleanup=True):

    
    if read_feature:
        print("Read Features")
        only_satellite = True
        img_features_query, ids_query, paths_query, img_features_gallery, ids_gallery, paths_gallery, coordinates_query, coordinates_gallery =  feature_read(read_path=save_path, device=config.device)
        if only_satellite:
            img_features_query, ids_query, paths_query = predict(config, model, query_loader, if_center)

    else:
        print("Extract Features:")
        if if_center:
            img_features_gallery, ids_gallery, paths_gallery, center_features_gallery = predict(config, model, gallery_loader, if_center)
            img_features_query, ids_query, paths_query, center_features_query = predict(config, model, query_loader, if_center)
        else:
            img_features_gallery, ids_gallery, paths_gallery = predict(config, model, gallery_loader, if_center)
            img_features_query, ids_query, paths_query = predict(config, model, query_loader, if_center)

        coordinates_gallery =  gallery_loader.dataset.coordinates
        coordinates_query = query_loader.dataset.coordinates

        if save_feature:
            print("Save Features")
            feature_save(img_features_query, ids_query, paths_query, img_features_gallery, ids_gallery, paths_gallery, coordinates_query, coordinates_gallery, save_path=save_path)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    indexOfTopK_list = []


    # Semantic-space clustering
    if if_aggregation:
        aggregation_path = save_path + "/kmeans_results.npz"
        if os.path.exists(aggregation_path):
            data = np.load(aggregation_path)
            cluster_labels = data["cluster_labels"]
            avg_features = torch.tensor(data["avg_features"], dtype=torch.float32).to(config.device)
        else:
            k = 1000
            sat_features = img_features_gallery.cpu().numpy()
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(sat_features)
            avg_features = kmeans.cluster_centers_
            avg_features = torch.tensor(avg_features, dtype=torch.float32).to(config.device)

            # np.savez(aggregation_path, cluster_labels=cluster_labels, avg_features=avg_features.cpu().numpy())

    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0

    center_num = 0

    for i in tqdm(range(len(ids_query))):
        # Semantic-space clustering and Geospatial aggregation
        if if_aggregation:

            query_feat = img_features_query[i]
            sim = avg_features @ query_feat
            sim = sim.cpu().numpy()

            topN = 200  
            topN_idx = np.argsort(sim)[-topN:][::-1]

            selected_indices_ordered = []

            for cluster_id in topN_idx:

                idx = np.where(cluster_labels == cluster_id)[0]
                selected_indices_ordered.extend(idx)

            selected_indices_ordered = np.array(selected_indices_ordered)
            selected_coords   = coordinates_gallery[selected_indices_ordered]

        else:
            selected_indices_ordered = np.arange(len(img_features_gallery))
            selected_coords = None

        selected_img_features_gallery = img_features_gallery[selected_indices_ordered] 
        selected_gl   = gl[selected_indices_ordered]
    
        data_tmp, index, score, good_index, junk_index, aggregation = eval_query(img_features_query[i], ql[i], selected_img_features_gallery, selected_gl, selected_coords)

        # Center-region matching
        if if_center:
            c_qf = center_features_query[i] 
            c_gf = center_features_gallery 
            delta_score = score[0] - score[1] 

            if c_qf is not None: 
                top_k = 2 
                region_size = 10 
                topk_index = list(selected_indices_ordered[index][:top_k]) 
                topk_c_gf = c_gf[topk_index, :] 

                if delta_score<0.03:
                    start = (28 - region_size) // 2 
                    end = start + region_size 
                    c_qf = c_qf.numpy() 
                    c_qf = np.mean(c_qf[:, start:end, start:end], axis=(1, 2)) 
                    c_qf = c_qf / np.linalg.norm(c_qf) 

                    distances = [] 
                    sims = [] 
                    coors = [] 

                    for j in range(top_k): 
                        coor, dist, sim = match_top_windows(c_qf, topk_c_gf[j], region_size=region_size) 
                        distances.append(dist) 
                        sims.append(-sim) 
                        coors.append(coor) 

                    sorted_pairs_sim = sorted(zip(sims, coors, distances, topk_index), key=lambda x: x[0]) 
                    sorted_index_sim = [p[3] for p in sorted_pairs_sim] 
                    sorted_pairs_dis = sorted(zip(sims, coors, distances, topk_index), key=lambda x: x[2]) 
                    sorted_index_dis = [p[3] for p in sorted_pairs_dis] 

                    if topk_index[0] != sorted_index_sim[0] and sorted_index_sim[0] == sorted_index_dis[0]:

                        H_start = sorted_pairs_sim[0][1][1] - region_size//2 
                        H_end = sorted_pairs_sim[0][1][1] + region_size//2 
                        W_start = sorted_pairs_sim[0][1][0] - region_size//2 
                        W_end = sorted_pairs_sim[0][1][0] + region_size//2 

                        m_gf2 = c_gf[sorted_index_sim[0]] 
                        m_gf2 = m_gf2.numpy() 
                        m_gf2 = np.mean(m_gf2[:, H_start: H_end, W_start: W_end],axis=(1, 2)) 
                        m_gf2 = m_gf2 / np.linalg.norm(m_gf2)

                        new_coor2, new_dist_2, _ = match_top_windows(m_gf2, topk_c_gf[0], region_size=region_size) 
                        H_start = new_coor2[1] - region_size//2 
                        H_end = new_coor2[1] + region_size//2 
                        W_start = new_coor2[0] - region_size//2 
                        W_end = new_coor2[0] + region_size//2 

                        m_gf4 = topk_c_gf[0] 
                        m_gf4 = m_gf4.numpy()
                        m_gf4 = np.mean(m_gf4[:, H_start: H_end, W_start: W_end],axis=(1, 2)) 
                        m_gf4 = m_gf4 / np.linalg.norm(m_gf4) 
                        
                        m_sim2 = np.dot(m_gf4, m_gf2) 

                        center_num += 1

                        if  m_sim2>0.7 and new_dist_2 > sorted_pairs_sim[0][2]:
                                
                            id = np.where(topk_index == sorted_index_sim[0])[0][0] 

                            front = index[index == index[id]] 
                            back = index[index != index[id]] 
                            index = np.concatenate((front, back)) 

                            data_tmp = compute_mAP(index, good_index, junk_index)     

        ap_tmp, CMC_tmp = data_tmp
        indexOfTopK_list.append(selected_indices_ordered[index[:100]])

        if CMC_tmp[0]==-1:
            continue

        CMC_tmp_full = np.zeros_like(CMC)
        CMC_tmp_full[:len(CMC_tmp)] = CMC_tmp
        CMC = CMC + CMC_tmp_full
        ap += ap_tmp

    with open(save_path + '/indexOfTopK_list.pkl', 'wb') as f:
        pickle.dump(indexOfTopK_list, f)
    print(f"save index: {save_path}")
    
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_gallery)*0.01)

    if if_print:
        string = []

        for i in ranks:
            string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))

        string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
        string.append('AP: {:.4f}'.format(AP))

        print(' - '.join(string))
    
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        #torch.cuda.empty_cache()
    
    return CMC[:20], AP


def eval_query(qf,ql,gf,gl,selected_coords=None):

    score = gf @ qf.unsqueeze(-1)
    
    score = score.squeeze().cpu().numpy()
 
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]    

    # good index
    query_index = np.where(np.any(gl == ql, axis=1))[0]

    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl==-1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)

    Geographic_aggregation = False

    # Geospatial aggregation
    if selected_coords is not None:

        topN_coord = selected_coords[index[:5]]

        topN_rad = np.radians(topN_coord)
        dist_matrix = haversine_distances(topN_rad) * 6371

        weighted_dist = dist_matrix[0]

        threshold = weighted_dist.mean() * 0.5
        mask = weighted_dist < threshold
        
        if mask.sum() > 2:
            num = 20
            all_rad = np.radians(selected_coords[index[:num]])
            center_rad = np.radians(topN_coord[0].reshape(1, -1))

            dist_to_center = haversine_distances(all_rad, center_rad).flatten() * 6371

            dist_norm = (dist_to_center - dist_to_center.min()) / (dist_to_center.max() - dist_to_center.min())

            rank = np.argsort(dist_norm)
            index[:num] = index[rank]

            CMC_tmp = compute_mAP(index, good_index, junk_index)

            Geographic_aggregation = True

    score_ = score[index[:100]]

    return CMC_tmp, index, score_, good_index, junk_index, Geographic_aggregation


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


def feature_save(img_features_query, ids_query, paths_query, img_features_gallery, ids_gallery, paths_gallery, coordinates_query, coordinates_gallery, save_path='./tmp_data'):
    img_features_query_np = img_features_query.cpu().numpy()
    ids_query_np = ids_query.cpu().numpy()
    paths_query_np = np.array(paths_query, dtype=object)

    img_features_gallery_np = img_features_gallery.cpu().numpy()
    ids_gallery_np = ids_gallery.cpu().numpy()
    paths_gallery_np = np.array(paths_gallery, dtype=object)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    savemat(save_path + '/features.mat', {
        'img_features_query': img_features_query_np,
        'ids_query': ids_query_np,
        'paths_query': paths_query_np,
        'img_features_gallery': img_features_gallery_np,
        'ids_gallery': ids_gallery_np,
        'paths_gallery': paths_gallery_np,
        'coordinates_query':  coordinates_query,
        'coordinates_gallery': coordinates_gallery,
    })

def feature_read(filepath='/features.mat', device='cuda', read_path='/tmp_data'):

    if not os.path.exists(read_path):
        os.makedirs(read_path)

    data = loadmat(read_path + filepath)

    img_features_query = torch.tensor(data['img_features_query'], device=device)
    ids_query = torch.tensor(data['ids_query'], device=device).squeeze()
    paths_query = data['paths_query'][0]
    coordinates_query = data['coordinates_query']

    img_features_gallery = torch.tensor(data['img_features_gallery'], device=device)
    ids_gallery = torch.tensor(data['ids_gallery'], device=device).squeeze()
    paths_gallery = data['paths_gallery'][0]
    coordinates_gallery = data['coordinates_gallery']

    return img_features_query, ids_query, paths_query, img_features_gallery, ids_gallery, paths_gallery, coordinates_query, coordinates_gallery

