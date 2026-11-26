#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :dsg-mbyolo_model_info.py
@Date        :2025/11/30 18:06:20
@Author      :Wying
@description :None
'''

import torch
import time
import argparse
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

def get_model_info(config_path, device='0', imgsz=640):
    """
    计算模型的 Params, GFLOPs 和 FPS
    """
    # 选择设备
    device = select_device(device)
    print(f"🚀 正在使用设备: {device}")

    # 1. 加载模型配置
    # 注意：这里使用 YAML 构建模型，不加载预训练权重，只看结构
    print(f"🔄 正在构建模型: {config_path} ...")
    try:
        model = YOLO(config_path)
    except Exception as e:
        print(f"❌ 模型构建失败。请确保你在 Mamba-YOLO 的项目根目录下运行此脚本，且环境已正确安装。")
        print(f"错误信息: {e}")
        return

    # 将模型移动到指定设备
    model_impl = model.model.to(device)
    model_impl.eval()

    # 2. 计算 Params 和 GFLOPs
    print("\n" + "="*40)
    print("📊 模型参数与计算量统计")
    print("="*40)
    # Ultralytics 自带的 info() 方法可以打印层数、参数量和 GFLOPs
    # detailed=True 会打印每一层的详细信息
    model.info(detailed=False, verbose=True)
    
    # 获取具体的数值 (如果不希望只看打印)
    n_p = sum(x.numel() for x in model_impl.parameters())  # number of parameters
    n_g = sum(x.numel() for x in model_impl.parameters() if x.requires_grad)  # gradients
    print(f"\n[自定义统计]")
    print(f"Params (M): {n_p / 1e6:.2f}")
    
    # 注意：对于自定义 Mamba 模块，标准的 thop 库可能无法精确计算 FLOPs。
    # Ultralytics 内部通常会尝试估算。如果显示 GFLOPs 为 0 或异常，通常是因为 Profiler 不认识 VSSBlock。
    
    # 3. 测试 FPS (推理速度)
    print("\n" + "="*40)
    print(f"⏱️  正在测试推理速度 (FPS) - 输入尺寸: {imgsz}x{imgsz}")
    print("="*40)
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    
    # 预热 (Warmup)
    print("🔥 正在预热 GPU...")
    with torch.no_grad():
        for _ in range(20):
            model_impl(dummy_input)
            
    # 正式测试
    test_runs = 100
    print(f"🏃 开始运行 {test_runs} 次推理...")
    
    torch.cuda.synchronize() if device.type != 'cpu' else None
    t_start = time.time()
    
    with torch.no_grad():
        for _ in range(test_runs):
            model_impl(dummy_input)
            
    torch.cuda.synchronize() if device.type != 'cpu' else None
    t_end = time.time()
    
    total_time = t_end - t_start
    avg_time = total_time / test_runs
    fps = 1 / avg_time
    
    print(f"总耗时: {total_time:.4f}s")
    print(f"单张耗时: {avg_time * 1000:.2f}ms")
    print(f"FPS: {fps:.2f}")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 你的 YAML 配置文件路径，例如 'ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml'
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='path to model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    
    opt = parser.parse_args()
    
    get_model_info(opt.config, opt.device, opt.imgsz)