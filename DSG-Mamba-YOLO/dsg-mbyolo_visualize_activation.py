#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :dsg-mbyolo_visualize_activation.py
@Date        :2025/11/30 18:36:37
@Author      :Wying
@description :None
'''

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# 配置部分：根据你的YAML结构，选择想要可视化的层索引
# Index 8  -> SPPF (Backbone的最后一层，通常包含最丰富的语义信息)
# Index 20 -> XSSBlock (Head部分，接近最终输出)
# -----------------------------------------------------------------------
TARGET_LAYER_INDEX = 5  # 可根据需要修改此索引 

class ActivationVisualizer:
    def __init__(self, model_path, device='cpu'):
        print(f"🔄 正在加载模型: {model_path} ...")
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model
        self.device = device
        self.model.to(self.device).eval()
        
        self.activations = None
        self._register_hook()

    def _hook_fn(self, module, input, output):
        """钩子函数：捕获层的输出"""
        self.activations = output

    def _register_hook(self):
        """注册钩子到指定的层"""
        # model.model.model 是 nn.Sequential 列表
        try:
            target_layer = self.model.model[TARGET_LAYER_INDEX]
            print(f"🎯 已挂载钩子到层索引 [{TARGET_LAYER_INDEX}]: {target_layer.__class__.__name__}")
            target_layer.register_forward_hook(self._hook_fn)
        except IndexError:
            print(f"❌ 错误：层索引 {TARGET_LAYER_INDEX} 超出范围。请检查 yaml 配置。")
            exit()

    def process_image(self, img_path, imgsz=640):
        # 1. 读取和预处理图像
        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            print(f"❌ 无法读取图像: {img_path}")
            return
        
        # 保持原始比例resize以便后续叠加
        h, w = img_raw.shape[:2]
        
        # 2. 准备模型输入
        # 使用 Ultralytics 的预处理流程太复杂，这里手动进行简单的 resize 和 normalize
        img_in = cv2.resize(img_raw, (imgsz, imgsz))
        img_in = img_in[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img_in = np.ascontiguousarray(img_in)
        img_in = torch.from_numpy(img_in).to(self.device)
        img_in = img_in.float() 
        img_in /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_in.ndimension() == 3:
            img_in = img_in.unsqueeze(0)

        # 3. 前向推理 (触发钩子)
        with torch.no_grad():
            _ = self.model(img_in)

        # 4. 处理激活图
        if self.activations is None:
            print("❌ 未捕获到激活图，请检查层索引是否正确。")
            return

        feat_map = self.activations  # shape: [1, C, H_feat, W_feat]
        
        # 这里的处理逻辑：
        # 对通道维度(dim=1)求平均，压缩成 [1, 1, H, W]
        # 这样可以看到所有特征通道共同关注的区域
        heatmap = torch.mean(feat_map, dim=1).squeeze()
        
        # ReLU 过滤负值 (可选，有时候保留负值也有意义，但通常热力图关注正激活)
        heatmap = torch.maximum(heatmap, torch.tensor(0., device=self.device))
        
        # 归一化到 0-1
        heatmap /= torch.max(heatmap) + 1e-10
        
        # 转回 numpy
        heatmap = heatmap.cpu().numpy()

        # 5. 生成可视化结果
        # 将热力图 resize 回原始图像大小
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # 转换为 RGB 热力图 (0-255)
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # 叠加到原图
        # alpha 控制热力图透明度
        alpha = 0.5
        overlay = cv2.addWeighted(img_raw, 1 - alpha, heatmap_colored, alpha, 0)

        return img_raw, heatmap_colored, overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./output_dir/corn/mambayolo_GBC_300/weights/best.pt')
    # parser.add_argument('--source', type=str, default='../../2024_code/datasets/images/activate_images/2023_09_23_16_02_12_140_950_1381_0_203.jpg', help='输入图像路径')
    parser.add_argument('--source', type=str, default='../../2024_code/datasets/images/activate_images/2023_09_23_16_02_12_140_950_1381_0_205.jpg', help='输入图像路径')
    # parser.add_argument('--source', type=str, default='../../2024_code/datasets/images/activate_images/2023_09_23_16_05_14_441_1330_1356_0_2030.jpg', help='输入图像路径')
    # parser.add_argument('--source', type=str, default='../../2024_code/datasets/images/activate_images/2023_09_24_11_25_40_497_950_1298_0_203.jpg', help='输入图像路径')
    
    parser.add_argument('--output', type=str, default='output_dir/activations_0_GBC_300', help='保存目录')
    parser.add_argument('--device', default='0', help='cpu 或 0')
    opt = parser.parse_args()

    # 创建保存目录
    save_dir = Path(opt.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化
    device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    visualizer = ActivationVisualizer(opt.weights, device=device)

    # 处理
    img_name = Path(opt.source).stem
    print(f"🖼️  正在处理: {opt.source}")
    
    orig, heatmap, overlay = visualizer.process_image(opt.source)

    if orig is not None:
        # 保存结果
        cv2.imwrite(str(save_dir / f"{img_name}_1_original.jpg"), orig)
        cv2.imwrite(str(save_dir / f"{img_name}_2_heatmap.jpg"), heatmap)
        cv2.imwrite(str(save_dir / f"{img_name}_3_overlay.jpg"), overlay)
        
        print(f"✅ 结果已保存至: {save_dir}")
        print(f"   - {img_name}_2_heatmap.jpg (纯热力图)")
        print(f"   - {img_name}_3_overlay.jpg (叠加图)")

if __name__ == "__main__":
    main()