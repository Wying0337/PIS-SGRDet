#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/3/31 17:25
# @Author  : WYing
# @Site    : 
# @File    : variant_temple_matching.py

import copy
import cv2
from PIL import Image
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def read_json(json_file):
    with open(json_file, 'r') as f:
        load_dict = json.load(f)
    f.close()
    return load_dict

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def matchTemplate(img1, img2):
    height, width = img1.shape[:2]
    # Calculate the coordinates of the top-left and bottom-right corners of the cropping region.
    center_x, center_y = width // 2, height // 2
    scale_factors = [150]
    all_matches = []
    for scale in scale_factors:
        crop_width, crop_height = scale, scale   # 150 150
        top_left_x = max(0, center_x - crop_width // 2)
        top_left_y = max(0, center_y - crop_height // 2)
        bottom_right_x = min(width, center_x + crop_width // 2)
        bottom_right_y = min(height, center_y + crop_height // 2)

        # crop the image
        template = img1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return max_val


def validation_img_match(similarity_results, similarity_name_sim, path_0, path_next, images0_list, images_cur_list):
    score_total = []
    for result in similarity_results:
        index_0 = result[0][0]
        index_1 = result[0][1]
        sim = result[1]
        map_0 = read_json(path_0)   # the current json_map
        map_next = read_json(path_next)   # the next json_map
        global_dict = map_0
        global_dict = {key: [value] for key, value in global_dict.items()}
        # The highest-scoring image pair stored in result[0],
        # Validate the mean similarity score for all matched patches post-coordinate-update.
        name_0 = images0_list[index_0]
        name_cur = images_cur_list[index_1]
        dict_0 = {value: eval(key) for key, value in map_0.items()}
        dict_next = {value: eval(key) for key, value in map_next.items()}
        cor_0 = dict_0[name_0]
        cor_next = dict_next[name_cur]
        # Update the coordinates in the second map to match those in the first map.
        copied_dict = copy.deepcopy(dict_next)
        for key, value in dict_next.items():
            _value = tuple(a - b for a, b in zip(value, cor_next))
            new_value = tuple(a + b for a, b in zip(_value, cor_0))
            copied_dict[key] = new_value  # name -> cor
        copy_next_map = {str(value): key for key, value in copied_dict.items()}   # cor -> name
        for key, value in copy_next_map.items():
            # If the key exists in the original dictionary, add it to the list.
            if key in global_dict:
                global_dict[key].append(value)
            else:
                # If the key does not exist in the original dictionary, add the key-value pair directly.
                global_dict[key] = [value]
        global_dict = {str(key): value for key, value in global_dict.items()}
        d = {tuple(term[0]): term[1] for term in similarity_name_sim}
        score = []
        for key, value in global_dict.items():
            if tuple(value) in d.keys():
                score.append(d.get(tuple(value)))
        score_avg = sum(score)/len(score)
        # print(f'score: {score},\nlength: {len(score)},\nscore_avg:{score_avg}')
        score_total.append(score_avg)
    # Find the index of the first occurrence of the maximum value and return the similarity_results
    score_total_ = np.array(score_total)
    score_total_ = np.where(score_total_ <= 0.5, 0, score_total_)
    unique_elements, counts = np.unique(score_total_, return_counts=True)
    products = unique_elements * counts
    max_product_index = np.argmax(products)
    element = unique_elements[max_product_index]
    index = np.where(score_total == element)[0][0]
    return similarity_results[index], index

def registration(num, json_dict_global, path_0, path_next):
    path_ahead = os.path.basename(path_0)[:-5]
    image_path_0 = os.path.join('./dish_processed', path_ahead)
    path_cur = os.path.basename(path_next)[:-5]
    image_path_cur = os.path.join('./dish_processed', path_cur)

    # dish image path list
    images0_l = os.listdir(image_path_0)
    images_cur_l = os.listdir(image_path_cur)
    images0_list = [os.path.join(image_path_0, name) for name in images0_l]
    images_cur_list = [os.path.join(image_path_cur, name) for name in images_cur_l]

    # load image from the folder, obtain the image path list 
    images0 = load_images_from_folder(image_path_0)
    images_cur = load_images_from_folder(image_path_cur)

    # calculate the similatiry
    similarity_results = []
    similarity_name_sim = []
    for i in range(len(images0)):
        for j in range(len(images_cur)):
            similarity = matchTemplate(images0[i], images_cur[j])
            similarity_results.append(((i, j), similarity))
            similarity_name_sim.append(((images0_list[i], images_cur_list[j]), similarity))
            # show the process
            print(f'\r{i*len(images_cur)+j+1}/{len(images0)*len(images_cur)}', end='')
    print()
    # Sort by similarity value in descending order.
    similarity_results.sort(key=lambda x: x[1], reverse=True)
    similarity_name_sim.sort(key=lambda x: x[1], reverse=True)

    # Iterate through the matching results in descending order
    similarity_results, index_record = validation_img_match(similarity_results, similarity_name_sim, path_0, path_next, images0_list, images_cur_list)
    file = f'./index_record.txt'
    with open(file, 'a', encoding='utf-8') as file:
        file.write(str(num) + ':' + str(index_record) + '\n')
    file.close()
    # save the json result
    dict_a = []
    for result in [similarity_results]:
        index = result[0]
        index_0 = result[0][0]
        index_1 = result[0][1]
        sim = result[1]
        x = ('cor', 'img_path', 'score')
        y = ((index_0, index_1), (images0_list[index_0], images_cur_list[index_1]), sim)
        d = dict.fromkeys(x, 0)
        for k, v in zip(x, y):
            d[k] = v
        dict_a.append(d)
    os.makedirs('./image_matches/', exist_ok=True)
    with open(f'./image_matches/{path_ahead}.json', 'w') as f:
        f.write(json.dumps(dict_a, indent=4))
        f.close
    print()

    # show result
    for result in [similarity_results]:
        (img_index1, img_index2), similarity = result
        # print(
            # f"Images {img_index1} and {img_index2} have a similarity score of {similarity}")

        img_01 = images0[img_index1][:, :, ::-1]
        img_02 = images_cur[img_index2][:, :, ::-1]
        plt.subplot(1, 2, 1)
        plt.imshow(img_01)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_02)
        plt.axis('off')
        # plt.show()
        os.makedirs('./Match/', exist_ok=True)
        p = f'./Match/{path_ahead}/'
        if not os.path.exists(p):
            os.makedirs(p)
        plt.savefig(f'./Match/{path_ahead}/{similarity}_{img_index1}-{img_index2}.jpg')


if __name__ == '__main__':
    # ------------------------------------
    os.chdir('./match_results/')
    print(os.getcwd())
    # ------------------------------------

    json_dirs = os.listdir('./json_map')
    json_paths = [os.path.join('./json_map', name) for name in json_dirs]
    json_paths.sort(key=lambda x: int(os.path.basename(x)[:19]))
    json_dict_global = read_json(json_paths[0])


    for i in range(0, len(json_dirs)-1):   # range(len(json_dirs)-1):
        print(json_paths[i])
        print(json_paths[i+1])
        registration(i, json_dict_global, json_paths[i], json_paths[i+1])