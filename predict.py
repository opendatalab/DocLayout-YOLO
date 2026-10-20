#!/usr/bin/env python3
"""Run local DocLayout-YOLO inference on an image, directory, or video."""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path,help='Existing local .pt checkpoint.')
    p.add_argument('--source',required=True,help='Image, directory, video, or stream source.')
    p.add_argument('--imgsz',type=int,default=1600)
    p.add_argument('--conf',type=float,default=0.25)
    p.add_argument('--iou',type=float,default=0.7)
    p.add_argument('--device',default='0')
    p.add_argument('--project',type=Path,default=Path('runs/predict'))
    p.add_argument('--name',default='doclayout')
    p.add_argument('--save-txt',action='store_true')
    return p.parse_args()

def main():
    a=parse_args(); model_path=a.model.expanduser().resolve()
    if not model_path.is_file() or model_path.suffix.lower()!='.pt':
        raise FileNotFoundError(f'--model must point to an existing local .pt file: {model_path}')
    YOLO(str(model_path)).predict(source=a.source,imgsz=a.imgsz,conf=a.conf,iou=a.iou,
        device=a.device,project=str(a.project.expanduser()),name=a.name,save=True,save_txt=a.save_txt)

if __name__=='__main__': main()
