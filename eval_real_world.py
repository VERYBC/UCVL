import os
import json
import torch
import subprocess
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.visloc import VislocDatasetEval, get_transforms
from sample4geo.evaluate.visloc import evaluate
from sample4geo.model import TimmModel_mix


@dataclass
class Configuration:
    # Model
    model: str = "BEMN-Tiny"  # 'BEMN-Tiny'|'BEMN-Base'

    # Override model image size
    img_size: int = 392

    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1  # -1 for all or int

    # Dataset
    dataset: str = 'D2S'
    dataset_name: str = 'Real-world filght data'
    data_folder: str = "./data"

    test_mode: str = ''  # '1_1' | ''

    # Test list
    TEST_LIST = [1, 2, 3]

    query_folder_test = []
    gallery_folder_test = []

    # Checkpoint to start from
    # "./checkpoints/BEMN-Tiny.pth"
    # "./checkpoints/BEMN-Base.pth"
    checkpoint_start = "./checkpoints/BEMN-Tiny.pth"

    # Retrieval methods
    if_SC_GA = True   # Semantic-space clustering and Geospatial aggregation
    if_CM = True      # Center-region matching method, it requires a significant amount of CPU memory

    # save and read features
    save_feature: bool = True # To output the positioning result, it needs to be set to True
    read_feature: bool = False

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------#
# Config                                                                      #
# -----------------------------------------------------------------------------#

config = Configuration()

if config.dataset == 'D2S':
    for i in config.TEST_LIST:
        config.query_folder_test.append(f'{config.data_folder}/{config.dataset_name}/{i:02}/drone_392')
        config.gallery_folder_test.append(
            f'{config.data_folder}/{config.dataset_name}/{i:02}/satellite_test' + config.test_mode)

elif config.dataset == 'S2D':
    for i in config.TEST_LIST:
        config.query_folder_test.append(
            f'{config.data_folder}/{config.dataset_name}/{i:02}/satellite_test' + config.test_mode)
        config.gallery_folder_test.append(f'{config.data_folder}/{config.dataset_name}/{i:02}/drone_392')

if __name__ == '__main__':

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))

    model = TimmModel_mix(config.model,pretrained=True)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    # load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    RECALLS = []
    APS = []
    QUERYS_NUM = []
    DISETANCES = []
    SDMS = []
    ACCURACYS = []

    for i in range(len(config.TEST_LIST)):

        print("\n{}Test {}-th scenario{}".format(20 * "-", config.TEST_LIST[i], 20 * "-"))

        # Reference Satellite Images
        query_dataset_test = VislocDatasetEval(data_folder=[config.query_folder_test[i]],
                                               transforms=val_transforms,
                                               )

        query_dataloader_test = DataLoader(query_dataset_test,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

        # Query Ground Images Test
        gallery_dataset_test = VislocDatasetEval(data_folder=[config.gallery_folder_test[i]],
                                                 transforms=val_transforms,
                                                 gallery_n=config.eval_gallery_n,
                                                 )

        gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                             batch_size=config.batch_size,
                                             num_workers=config.num_workers,
                                             shuffle=False,
                                             pin_memory=True)

        print("Query Images Test:", len(query_dataset_test))
        print("Gallery Images Test:", len(gallery_dataset_test))

        RECALL, AP = evaluate(config=config,
                              model=model,
                              query_loader=query_dataloader_test,
                              gallery_loader=gallery_dataloader_test,
                              ranks=[1, 2, 5],
                              save_feature=config.save_feature,
                              read_feature=config.read_feature,
                              if_center=config.if_CM,
                              if_aggregation=config.if_SC_GA,
                              save_path='./tmp_data',
                              cleanup=True)

        RECALLS.append(RECALL)
        APS.append(AP)
        QUERYS_NUM.append(len(query_dataset_test))

        print("\n Calculate the distance:")
        if config.if_CM:
            subprocess.run([
                "python", "evaluateDistance.py",
                "--root_dir", f'{config.data_folder}/{config.dataset_name}/',
                "--TEST_LIST", str(config.TEST_LIST[i]),
                "--if_center",
                "--if_print",
                '--read_index',
                '--model_path', './tmp_data'
            ])
        else:
            subprocess.run([
                "python", "evaluateDistance.py",
                "--root_dir", f'{config.data_folder}/{config.dataset_name}/',
                "--TEST_LIST", str(config.TEST_LIST[i]),
                "--if_print",
                '--read_index',
                '--model_path', './tmp_data'
            ])

        DISETANCE = json.load(open("./tmp_data/MA@K_200.json"))
        SDM = json.load(open("./tmp_data/SDM@K(1,10).json"))
        ACCURACY = [v for k, v in DISETANCE.items() if k not in ["TOP1_Distance", "TOP5_Distance"]]

        DISETANCES.append([DISETANCE['TOP1_Distance'], DISETANCE['TOP5_Distance']])
        SDMS.append(SDM['1'])
        ACCURACYS.append(ACCURACY)

    print("\n Calculate the average indicator:\n")

    RECALLS = np.array(RECALLS)
    APS = np.array(APS)
    QUERYS_NUM = np.array(QUERYS_NUM)
    DISETANCES = np.array(DISETANCES)
    SDMS = np.array(SDMS)
    ACCURACYS = np.array(ACCURACYS)

    weights = np.array(QUERYS_NUM) / np.sum(QUERYS_NUM)  # shape=(num_segments,)

    # weighted average
    avg_recall = np.average(RECALLS, axis=0, weights=weights)
    avg_AP = np.average(APS, axis=0, weights=weights)
    avg_distance = np.average(DISETANCES, axis=0, weights=weights)
    avg_sdm = np.average(SDMS, axis=0, weights=weights)
    avg_accuracy = np.average(ACCURACYS, axis=0, weights=weights)

    string = []

    for i in [1, 5, 10]:
        string.append('Avg_Recall@{}: {:.4f}'.format(i, avg_recall[i - 1] * 100))

    string.append('Avg_AP: {:.4f}'.format(avg_AP))

    print(' - '.join(string))

    # print("\nSDM@{} = {:.2f}%".format('1', avg_sdm * 100))

    print("\nTOP1_Distance = {:.2f} m   ".format(avg_distance[0]))
    print("\nTOP5_Distance = {:.2f} m \n".format(avg_distance[1]))

    keys = [k for k in DISETANCE.keys() if k not in ["TOP1_Distance", "TOP5_Distance"]]
    MA_average = {k: float(v) for k, v in zip(keys, avg_accuracy)}

    save_path = "./tmp_data/MA@K_average.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(MA_average, f, indent=4)

    print(f"Save the MA@K_average: {save_path}")
