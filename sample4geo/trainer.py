import time
import torch
import pickle
import os
from torch.autograd import Variable
from tqdm import tqdm
from .utils import AverageMeter
from torch.amp import autocast
import torch.nn.functional as F

def predict(train_config, model, dataloader, if_center=False, if_denseuav=False):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    center_features_list = []
    ids_list = []
    paths_list = []

    times = []

    with torch.no_grad():
        for img, ids, path in bar:

            this1 = time.time()
        
            ids_list.append(ids)
            paths_list.append(path)
   
            with autocast(device_type='cuda'): # cuda

                if if_center:
                    img = img.to(train_config.device)
                    img_feature, center_feature = model(img, center_feature=True)
                elif if_denseuav:
                    for i in range(2):
                        if(i==1):
                            img = fliplr(img)
                        input_img = Variable(img.cuda())
                        img_feature, _ = model(input_img, None)
                        img_feature = img_feature[1]
                        if i==0:
                            ff = img_feature
                        else:
                            ff += img_feature
                        # norm feature
                        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                        ff = ff.div(fnorm.expand_as(ff))
                    img_feature = ff
                    center_feature = torch.zeros(1)
                else:
                    img = img.to(train_config.device)
                    img_feature = model(img)
                    center_feature = torch.zeros(1)
            
                # normalize is calculated in fp32
                if train_config.normalize_features and not if_denseuav:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
            center_features_list.append(center_feature.to(torch.float32).cpu())

            this3 = time.time()
            times.append(this3 - this1)

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        center_features = torch.cat(center_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        paths_list = [item for sublist in paths_list for item in sublist]

        print('Cost time is %.4f seconds \n' % (sum(times) / len(times)))

    if train_config.verbose:
        bar.close()
    
    if if_center:
        return img_features, ids_list, paths_list, center_features
    else:
        return img_features, ids_list, paths_list
    

def fliplr(img):

    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip