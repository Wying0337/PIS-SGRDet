#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :dsg-mbyolo_val.py
@Date        :2025/12/04 09:38:17
@Author      :Wying
@description :None
'''


from ultralytics import YOLO

model = YOLO('./output_dir/corn/mambayolo_GBC_300/weights/best.pt')  # load a custom model
results = model.val(data='./ultralytics/cfg/datasets/corn_data.yaml', 
                    batch=8, imgsz=640, device='0', 
                    iou=0.35, 
                    conf=0.3, 
                    project= './runs/detect/',
                    name='200epoch_GBC',
                    workers=8)  # validate the model
