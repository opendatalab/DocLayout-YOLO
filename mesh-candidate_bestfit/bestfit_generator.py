import os
import json
import time
import torch
import random
import datetime
import argparse
import itertools
import torchvision
import multiprocessing
from utils.process import *

random.seed(datetime.datetime.now().timestamp())


def bestfit_generator(element_all):
    """
    Apply the Mesh-candidate Bestfit algorithm to generate diverse layouts.

    Args:
        element_all (dict): Loaded elements from dataset json file.
        output_dir (str): Directory to save the generated layouts.
    """
    # Default candidate_num = 500
    candidate_num = 500
    large_elements_idx = random.sample(list(range(len(element_all['large']))), int(candidate_num*0.99))
    small_elements_idx = random.sample(list(range(len(element_all['small']))), int(candidate_num*0.01))
    cand_elements = [element_all['large'][large_idx] for large_idx in large_elements_idx] + [element_all['small'][small_idx] for small_idx in small_elements_idx]

    # Initially, randomly put an element
    put_elements = []
    e0 = random.choice(cand_elements)
    cx = random.uniform(min(e0.w/2, 1-e0.w/2), max(e0.w/2, 1-e0.w/2))
    cy = random.uniform(min(e0.h/2, 1-e0.h/2), max(e0.h/2, 1-e0.h/2))
    e0.cx, e0.cy = cx, cy
    put_elements = [e0]
    cand_elements.remove(e0)
    small_cnt = 1 if e0.w < 0.05 or e0.h < 0.05 else 0

    # Iterativelly insert elements
    while True:
        # Construct meshgrid based on current layout
        put_element_boxes = []
        xticks, yticks = [0,1], [0,1]
        for e in put_elements:
            x1, y1, x2, y2 = e.cx-e.w/2, e.cy-e.h/2, e.cx+e.w/2, e.cy+e.h/2
            xticks.append(x1)
            xticks.append(x2)
            yticks.append(y1)
            yticks.append(y2)
            put_element_boxes.append([x1, y1, x2, y2])
        xticks, yticks = list(set(xticks)), list(set(yticks))
        pticks = list(itertools.product(xticks, yticks))
        meshgrid = list(itertools.product(pticks, pticks))
        put_element_boxes = torch.Tensor(put_element_boxes)

        # Filter out invlid grids
        meshgrid = [grid for grid in meshgrid if grid[0][0] < grid[1][0] and grid[0][1] < grid[1][1]]
        meshgrid_tensor = torch.Tensor([p1 + p2 for p1, p2 in meshgrid])
        iou_res = torchvision.ops.box_iou(meshgrid_tensor, put_element_boxes)
        valid_grid_idx = (iou_res.sum(dim=1) == 0).nonzero().flatten().tolist()
        meshgrid = meshgrid_tensor[valid_grid_idx].tolist()

        # Search for the Mesh-candidate Bestfit pair
        max_fill, max_grid_idx, max_element_idx = 0, -1, -1
        for element_idx, e in enumerate(cand_elements):
            for grid_idx, grid in enumerate(meshgrid):
                if e.w > grid[2] - grid[0] or e.h > grid[3] - grid[1]:
                    continue
                element_area = e.w * e.h
                grid_area = (grid[2] - grid[0]) * (grid[3] - grid[1])
                if element_area/grid_area > max_fill:
                    max_fill = element_area/grid_area
                    max_grid_idx = grid_idx
                    max_element_idx = element_idx

        # Termination condition
        if max_element_idx == -1 or max_grid_idx == -1:
            break
        else:
            maxfit_element = cand_elements[max_element_idx]
            if maxfit_element.w < 0.05 or maxfit_element.h < 0.05:
                small_cnt += 1
            if small_cnt > 5:
                break
            else:
                pass

        # Put the candidate to the center of the grid
        cand_elements.remove(maxfit_element)
        maxfit_element.cx = (meshgrid[max_grid_idx][0] + meshgrid[max_grid_idx][2])/2
        maxfit_element.cy = (meshgrid[max_grid_idx][1] + meshgrid[max_grid_idx][3])/2
        put_elements.append(maxfit_element)

    # Apply a rescale transform to introduce more diversity
    for _, e in enumerate(put_elements):
        e.gen_real_bbox()
    layout = Layout(cand_elements=put_elements)

    # Convert the layout to json file format
    boxes, categories, relpaths = [], [], []
    for element in layout.cand_elements:
        cx, cy, w, h = element.get_real_bbox()
        x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
        boxes.append([x1, y1, x2, y2])
        categories.append(element.category-1) # Exclude the "__background__" category (category_id = 0)
        relpaths.append(element.filepath)

    output_layout = {
        "boxes": boxes,
        "categories": categories,
        "relpaths": relpaths
    }

    # To prevent process blocking, save the result of each layout in a timely manner.
    with open(os.path.join(OUTPUT_DIR,str(time.time()).replace(".", "_")+'.json'),'w') as f:
        json.dump(output_layout, f)
        
    return output_layout



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_num', default=None, required=True, type=int, help='number of layouts to generate')
    parser.add_argument('--n_jobs', default=None, required=True, type=int, help='number of processes to use in multiprocessing')
    parser.add_argument('--json_path', default=None, required=True, type=str, help='original json file of the dataset')
    parser.add_argument('--output_dir', default='./generated_layouts/seperate', type=str, help='output directory of generated seperate layouts')
    args = parser.parse_args()
    
    element_all = read_data(args.json_path)
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    # Using multiprocessing to accelerate generation
    n_jobs = args.n_jobs
    with multiprocessing.Pool(n_jobs) as p:
        generated_layout = p.starmap(
            bestfit_generator, [(element_all,) for _ in range(args.generate_num)]
        )
    p.close()
    p.join()