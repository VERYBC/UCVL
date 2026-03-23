import torch
import timm
import random
import cv2
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

args_do_not_overide = ['verbose', 'resume_from']
os.environ["TORCH_CUDA_DSA_DISABLE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class TimmModel_mix(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=True,
                 ):  

        super(TimmModel_mix, self).__init__()

        self.conv_pool  = torch.nn.AdaptiveAvgPool2d(1)
        self.dino_pool  = torch.nn.AdaptiveAvgPool1d(1)
        self.model_name = model_name
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if "Tiny" in model_name:
            self.model1 = timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k_384', pretrained=pretrained, num_classes=0)
            # self.model2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            # local loading
            self.model2 = torch.hub.load(current_dir + '/model_lib/dinov2/facebookresearch_dinov2_main/','dinov2_vits14_reg', trust_repo=True, source='local')
            self.model2.load_state_dict(torch.load(current_dir + '/model_lib/dinov2/dinov2_vits14_reg4_pretrain.pth'))
        elif "Base" in model_name:
            self.model1 = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=pretrained, num_classes=0)
            # self.model2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            # local loading
            self.model2 = torch.hub.load(current_dir + '/model_lib/dinov2/facebookresearch_dinov2_main/','dinov2_vitb14_reg', trust_repo=True, source='local')
            self.model2.load_state_dict(torch.load(current_dir + '/model_lib/dinov2/dinov2_vitb14_reg4_pretrain.pth'))

        self.region_size = 10
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
  
    def get_config(self, ):
        data_config1 = timm.data.resolve_model_data_config(self.model1)
        return data_config1

    def set_grad_checkpointing(self, enable=True):

        self.model1.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None, center_feature=False):
        
        if img2 is not None:

            #------------------------------------------------------------img1
            # convnextv2
            x = self.model1.stem(img1)
            feat1_1 = self.model1.stages(x)
            local_feat1 = self.conv_pool(feat1_1).squeeze(-2, -1)

            # dino
            x = self.model2.prepare_tokens_with_masks(img1)
            for blk in self.model2.blocks:
                x = blk(x)
            x_norm = self.model2.norm(x)
            ret = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.model2.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1 :],
                "x_prenorm": x,
            }
            feat1_2 = self.model2.head(ret["x_norm_patchtokens"])

            global_feat1 = self.dino_pool(feat1_2.permute(0, 2, 1)).squeeze(-1)

            # Conv
            c1 = local_feat1

            # Dino
            d1 = global_feat1

            # ------------------------------------------------------------img2
            # convnext
            x = self.model1.stem(img2)
            feat2_1 = self.model1.stages(x)
            local_feat2 = self.conv_pool(feat2_1).squeeze(-2, -1)

            # dino
            x = self.model2.prepare_tokens_with_masks(img2)
            for blk in self.model2.blocks:
                x = blk(x)
            x_norm = self.model2.norm(x)
            ret = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.model2.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1 :],
                "x_prenorm": x,
            }
            feat2_2 = self.model2.head(ret["x_norm_patchtokens"])

            global_feat2 = self.dino_pool(feat2_2.permute(0, 2, 1)).squeeze(-1)

            # Conv
            c2 = local_feat2

            # Dino
            d2 = global_feat2

            return  c1, c2, d1, d2

        else:
            # convnext
            x = self.model1.stem(img1)
            feat1 = self.model1.stages(x)
            local_feat = self.conv_pool(feat1).squeeze(-2, -1)

            # dino
            x = self.model2.prepare_tokens_with_masks(img1)
            for blk in self.model2.blocks:
                x = blk(x)
            x_norm = self.model2.norm(x)
            ret = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.model2.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1 :],
                "x_prenorm": x,
            }
            feat2 = self.model2.head(ret["x_norm_patchtokens"])
            global_feat = self.dino_pool(feat2.permute(0, 2, 1)).squeeze(-1)

            # fusion
            local_feat = F.normalize(local_feat, p=2, dim=-1)
            global_feat = F.normalize(global_feat, p=2, dim=-1)

            feat = torch.cat((local_feat, global_feat), dim=-1)

            if center_feature:

                # center feature
                width = feat2.shape[1] // 784

                feat2_ = feat2.permute(0, 2, 1)
                feat2_ = feat2_.view(feat2_.shape[0], feat2_.shape[1], 28, 28 * width)

                feat1_ = F.interpolate(feat1, size=(28, 28 * width), mode='bilinear', align_corners=False) 

                feat1_ = F.normalize(feat1_, p=2, dim=1)
                feat2_ = F.normalize(feat2_, p=2, dim=1)

                center_feat =  torch.cat([feat1_, feat2_], dim=1)

                return feat, center_feat
            else:
                return feat
