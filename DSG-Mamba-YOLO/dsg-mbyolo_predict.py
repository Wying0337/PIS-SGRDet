#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :dsg-mbyolo_predict.py
@Date        :2025/12/04 15:47:13
@Author      :Wying
@description :None
'''

from ultralytics import YOLO
model = YOLO('./output_dir/corn/mambayolo_GBC_MIOU/weights/best.pt')  # load a custom model
results = model.predict(source='/home/wying/Documents/2024_code/datasets/images/val',  # image, video, dir, webcam, stream
                        imgsz=640,
                        batch=8,
                        iou=0.35, 
                        conf=0.3, 
                        device='0',
                        save=True,
                        project='./runs/predict/',
                        name='mbyolo_infer_GBC_MIOU',
                        exist_ok=True,
                        line_width =1,
                        )  # predict on images and save results