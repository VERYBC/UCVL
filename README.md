#  <div align="center">UAV Coarse Visual Localization in Large-Scale Continuous Scenes</div>
This repository contains the dataset and the code for our paper **UAV Coarse Visual Localization in Large-Scale Continuous Scenes**. Thank you for your kindly attention.
## 1.XIAN-Visloc Dataset
We collect a total of **21 flight trajectories** in Xi’an and Weinan, Shaanxi, China, using DJI drones. The total flight distance is **81.13 km**, and the trajectories exhibit almost no overlap. Among them, 19 trajectories collected in Xi’an are used for training and testing, while the remaining 2 collected in Weinan are used exclusively for testing. In total, **9,871 UAV-view images** are captured, covering various representative urban and suburban environments. The collected UAV images have resolutions of **either 3840×2160 or 490×490**, and each image is associated with GPS measurements to enable meter-level evaluation. In addition, we extract 6 satellite maps from the latest Level-19 Google Maps tiles, each with a ground resolution of 0.247 m, covering all 21 flight trajectories. Among these, four maps correspond to Xi’an with a total coverage area of **59.52 km<sup>2**, and two correspond to Weinan with a total coverage area of **52.76 km<sup>2**.
![XIAN-Visloc](docs/XIAN-Visloc.png)
## 2.Constrution of the Satellite image library of XIAN-Visloc and UAV-Visloc
1. Download XIAN-Visloc and [UAV-Visloc](https://github.com/IntelliSensing/UAV-VisLoc?tab=readme-ov-file) .
2. Build training and testing sets using the .py files in **scripts** folder.
* Due to the large size of the constructed satellite image library, we recommend scaling the satellite image resolution to the required resolution in advance, that is, modifying the **target_size** in the file.
## 3.Test on XIAN-Visloc and UAV-Visloc
We will soon supplement the code ！

