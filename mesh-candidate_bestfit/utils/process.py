import os
import json
from .base import *

def read_data(json_file):
    """
    Load elements from dataset json file.

    Args:
        json_file (str): A dataset json file path with coco format.
    """
    data = json.load(open(json_file))
    category_id2name = {item['id']:item['name'] for item in data['categories']}
    element_all = {'large':[], "small":[]}
    image2anno = {image["id"]:image for image in data["images"]}
    for anno in data["annotations"]:
        H, W = image2anno[anno["image_id"]]["height"], image2anno[anno["image_id"]]["width"]
        w, h = anno["bbox"][2], anno["bbox"][3]
        if w/W < 0.01 or h/H < 0.01:
            continue
        anno_id, category_id = anno['id'], anno["category_id"]
        e = element(cx=None, cy=None, h=h/H, w=w/W, category=category_id,filepath=f'{category_id2name[category_id]}/{anno_id}.jpg')
        if w/W >= 0.05 and h/H >= 0.05:
            element_all['large'].append(e)
        else:
            element_all['small'].append(e)
    return element_all


def sample_hw(width_range, ratio_range, max_height):
    """
    Randomly sample a (w,h) size for rendering from a given range.

    Args:
        width_range (list): Given range of width.
        ratio_range (list): Given range of h/w ratio.
        max_height (int): Upper bound of height.
    """
    w = random.randint(width_range[0], width_range[1])
    ratio = random.uniform(ratio_range[0], ratio_range[1])
    h = min(max_height, int(w*ratio))
    return w, h