import os
import cv2
import torch
import argparse
from doclayout_yolo import YOLOv10

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--image-path', default=None, required=True, type=str)
    parser.add_argument('--res-path', default='outputs', required=False, type=str)
    parser.add_argument('--imgsz', default=1024, required=False, type=int)
    parser.add_argument('--line-width', default=5, required=False, type=int)
    parser.add_argument('--font-size', default=20, required=False, type=int)
    parser.add_argument('--conf', default=0.2, required=False, type=float)
    args = parser.parse_args()
    
    # Automatically select device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLOv10(args.model)  # load an official model

    det_res = model.predict(
        args.image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
    )
    annotated_frame = det_res[0].plot(pil=True, line_width=args.line_width, font_size=args.font_size)
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    output_path = os.path.join(args.res_path, args.image_path.split("/")[-1].replace(".jpg", "_res.jpg"))
    cv2.imwrite(output_path, annotated_frame)
    print(f"Result saved to {output_path}")