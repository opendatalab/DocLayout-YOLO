import os
import cv2
import argparse
from doclayout_yolo import YOLOv10

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--image-path', default=None, required=True, type=str)
    parser.add_argument('--device', default=None, required=True, type=str)
    parser.add_argument('--imgsz', default=1024, required=False, type=int)
    parser.add_argument('--line-width', default=5, required=False, type=int)
    parser.add_argument('--font-size', default=20, required=False, type=int)
    parser.add_argument('--conf', default=0.2, required=False, type=float)
    args = parser.parse_args()
    
    model = YOLOv10(args.model)  # load an official model

    det_res = model.predict(
        args.image_file,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )
    annotated_frame = det_res[0].plot(pil=True, line_width=args.line_width, font_size=args.font_size)
    cv2.imwrite(args.image_path.split("/")[-1].replace(".png", "_res.png"), annotated_frame)