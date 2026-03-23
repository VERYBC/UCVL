#  <div align="center">UAV Coarse Visual Localization in Large-Scale Continuous Scenes</div>
This repository contains the dataset and the code for our paper **UAV Coarse Visual Localization in Large-Scale Continuous Scenes**. Thank you for your kindly attention.
## 1. XIAN-Visloc Dataset
We collect a total of **21 flight trajectories** in Xi’an and Weinan, Shaanxi, China, using DJI drones. The total flight distance is **81.13 km**, and the trajectories exhibit almost no overlap. Among them, 19 trajectories collected in Xi’an are used for training and testing, while the remaining 2 collected in Weinan are used exclusively for testing. In total, **9,871 UAV-view images** are captured, covering various representative urban and suburban environments. The collected UAV images have resolutions of **either 3840×2160 or 490×490**, and each image is associated with GPS measurements to enable meter-level evaluation. In addition, we extract 6 satellite maps from the latest Level-19 Google Maps tiles, each with a ground resolution of 0.247 m, covering all 21 flight trajectories. Among these, four maps correspond to Xi’an with a total coverage area of **59.52 km<sup>2**, and two correspond to Weinan with a total coverage area of **52.76 km<sup>2**.
![XIAN-Visloc](docs/XIAN-Visloc.png)
## 2. Constrution of the Satellite image library of XIAN-Visloc and UAV-Visloc

1. Download [XIAN-Visloc](https://huggingface.co/datasets/VERYBC/XIAN_Visloc) and [UAV-Visloc](https://github.com/IntelliSensing/UAV-VisLoc?tab=readme-ov-file) .
   
3. Build train and test sets using the .py files in `scripts` folder.
   
📌 Due to the large size of the constructed satellite image library, we recommend scaling the satellite image resolution to the required resolution in advance, that is, modifying the `target_size` in the file.
  
## 3. Real-world flight data test
### 3.1. Data preprocess  

1. Download the [Real-world flight data](https://huggingface.co/datasets/VERYBC/XIAN_Visloc) and place it in the `data/` folder.

2. Construct the satellite reference library:
   - Run `scripts/Real_world_flight_test.py`. Set the satellite image resolution to **(392, 392)**.

3. Resize UAV images:
   - Run `scripts/UAV image rescaling.py`. The processed UAV data will be saved to the `drone_392/` folder.
     
📌 If you do not adopt the UAV image resolution of (392, 392), you will need to modify the file path in the subsequent code.
   
### 3.2. Coarse localization  

1. Retrieval results

   - Download the model weight files [[Google]](https://drive.google.com/drive/folders/1E2IXkT4bqmuNL_bT2Fo20K0c39E1pOCI?usp=drive_link) and place them in the `checkpoints/` directory.
     
   - Run the evaluation script: `python eval_real_world.py`, you will get:  

      | Method   | R@1     | R@5      | AP       | Dis@1    | Dis@5    |
      | -------- | ------- | -------- | -------- | -------- | -------- |
      | BEMN-T   | 55.10   | 74.30    | 43.15    | 888.03   | 431.71   |
      | BEMN-B   | 70.72   | 87.09    | 57.84    | 503.38   | 180.39   |
     
   - The retrieval results will be saved to `tmp_data`.
     
2. Coarse localization results
   
   - Run `Coarse localization_result.py`, you will get:  
     
      | ID   | Top 1 error (m)     | Top 1 error (m)  | Top1 lost fraction (%)     | Top5 lost fraction (%) |
      | -------- | ------- | -------- | -------- | -------- |
      | 1  | 61.02   | 50.96    | 21.67    | 6.67   |
      | 2  | 66.75   | 42.78    | 17.71    | 8.98   |
      | 3  | 48.26   | 36.17    | 14.03    | 10.86  |

### 3.3. Fine localization

   - Download the LoFTR model [weight files](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf) and place them in the `feature_match/LoFTR/weights` directory.

   - Run `Fine_localization.py`, the result will be saved to 'feature_match/results'.
     
   - Run `Fine_localization_result.py`, you will get:
     
      | Method   | Error (m)     | Success rate (%)  | 
      | -------- | ------- | -------- |
      | Top1+LightGlue  | 33.00   | 57.14    | 
      | Top1+LoFTR      | 28.30   | 61.90    | 
      | Top1+RoMa       | 18.06   | 91.67    |
      | Top1+RoMav2     | 14.47   | 95.24    |
     
   📌 The above results are obtained using an image resolution of 392 × 392. Using the original resolution will reproduce the results reported in the paper. The LoFTR model consistently uses a resolution of 1024.

   📌 The results of RoMa and RoMav2 may exhibit slight variations.
   
     

     
