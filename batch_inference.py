import os
import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from doclayout_yolo import YOLOv10

import pdb

def read_image(im):
    im = Image.open(im).convert("RGB")
    im = np.asarray(im)[:, :, ::-1]
    im = np.ascontiguousarray(im)  # contiguous
    return im

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--image-txt', default=None, required=True, type=str)
    parser.add_argument('--batch-size', default=128, required=False, type=str)
    parser.add_argument('--res-path', default='outputs', required=False, type=str)
    parser.add_argument('--imgsz', default=1024, required=False, type=int)
    parser.add_argument('--line-width', default=5, required=False, type=int)
    parser.add_argument('--font-size', default=20, required=False, type=int)
    parser.add_argument('--conf', default=0.2, required=False, type=float)
    parser.add_argument('--visualize', action='store_true', help="whether to visualize detection results")
    args = parser.parse_args()
    
    # Automatically select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLOv10(args.model)

    start = time.time()
    image_path = list(open(args.image_txt, "r").readlines())
    image_path = [image.strip() for image in image_path]
    with ThreadPoolExecutor(max_workers=100) as executor:
        image_list = list(executor.map(read_image, image_path))
    print(f"Read Image Time: {time.time() - start} seconds", )

    batch_size = args.batch_size
    start = time.time()
    for i in tqdm(range(0, len(image_list), batch_size)):
        image_batch = image_list[i:i+batch_size]
        path_batch = image_path[i:i+batch_size]
        det_res = model.predict(
            image_batch,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
        )
        
        if args.visualize:
            for res, path in zip(det_res, path_batch):
                img_ext = path.split(".")[-1]
                annotated_frame = res.plot(pil=True, line_width=args.line_width, font_size=args.font_size)
                if not os.path.exists(args.res_path):
                    os.makedirs(args.res_path)
                output_path = os.path.join(
                    args.res_path, 
                    path.strip().split("/")[-1].replace(f".{img_ext}", f"_res.{img_ext}")
                )
                cv2.imwrite(output_path, annotated_frame)
                
    end = time.time()
    fps = round(len(image_list)/(end - start), 2)
    print(f"Inference Time: {end - start} seconds, FPS: {fps}")


'''
python batch_inference.py \
--model doclayout_yolo_docstructbench_imgsz1024.pt \
--image-txt ../DocLayout-YOLO-dev2/layout_data/doclaynet/val.txt
'''