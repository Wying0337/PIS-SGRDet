#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :dsg-mbyolo_germination_rate.py
@Date        :2025/12/03 22:19:54
@Author      :Wying
@description :None
'''

import os
import json
from pathlib import Path
from ultralytics import YOLO


from pathlib import Path
from ultralytics import YOLO

def calculate_germination_rate(model_path, data_root):
    """
    计算发芽率，输出格式为: { "(1, 9)": [{"img.jpg": 0.88}, ...] }
    """
    # 1. 加载模型
    model = YOLO(model_path, task='detect')
    data_root = Path(data_root)
    results_dict = {}

    # 【优化1】预先获取 'bud' 的类别 ID，避免在循环中重复查找
    # 假设 model.names 是 {0: 'bud', 1: 'leaf'...}
    bud_id = next((k for k, v in model.names.items() if 'bud' in v.lower()), None)
    if bud_id is None:
        print("Warning: Model does not contain 'bud' class.")

    # 定义文件夹排序规则 (自然排序: 1_9 -> 2_4)
    def sort_key(p):
        try: return [int(x) for x in p.name.split('_')]
        except: return [float('inf')] # 格式不对排到最后

    # 2. 遍历文件夹
    # 过滤掉非文件夹项，并排序
    folders = sorted([f for f in data_root.iterdir() if f.is_dir()], key=sort_key)
    
    for folder in folders:
        print(f"Processing: {folder.name}")
        
        # 【优化2】直接通过字符串替换生成 Key: "2_4" -> "(2, 4)"
        folder_key = f"({folder.name.replace('_', ', ')})"
        current_images = []

        # 3. 遍历图片
        for img_path in sorted(folder.glob("*.jpg")):
            try:
                # 推理 (只取第一个结果对象)
                # verbose=False 关闭 YOLO 内部打印，减少刷屏
                result = model(str(img_path), conf=0.3, iou=0.4, verbose=False)[0]
                
                # 【优化3】利用 Tensor 快速统计，代替繁琐的 for 循环
                boxes = result.boxes
                total = len(boxes)
                
                # 如果检测到了目标且找到了bud的ID，计算bud数量
                # (boxes.cls == bud_id) 返回布尔矩阵，.sum() 计算 True 的个数
                bud_count = (boxes.cls == bud_id).sum().item() if (total > 0 and bud_id is not None) else 0
                
                # 计算比率
                rate = round(bud_count / total, 2) if total > 0 else 0.0
                
                # 存入列表 (符合新格式)
                current_images.append({img_path.name: rate})
                
                print(f"  {img_path.name}: {rate} ({bud_count}/{total})")

            except Exception as e:
                print(f"  Error on {img_path.name}: {e}")
                current_images.append({img_path.name: 0.0}) # 出错记录为0
        
        # 存入主字典
        results_dict[folder_key] = current_images

    return results_dict


def save_results_to_json(results, output_path):
    """
    Save results to JSON file
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    model_path = './output_dir/corn/mambayolo_GBC_300/weights/best.pt'
    data_root = '/home/wying/Documents/2024_code/sequence_data_50/'
    output_json_dir = './output_dir/germination_rates/'
    output_json_name = './germination_rate_results_mambayolo_GBC_300.json'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit(1)
    
    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {data_root}")
        exit(1)

    if not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir)
    
    print(f"Starting germination rate calculation...")
    print(f"Model: {model_path}")
    print(f"Data root: {data_root}")
    print()
    
    # Calculate germination rate
    results = calculate_germination_rate(model_path, data_root)
    
    # Save results
    output_json = os.path.join(output_json_dir, output_json_name)
    save_results_to_json(results, output_json)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics:")
    print("="*50)
    print(f"Total images processed: {len(results)}")
    
   