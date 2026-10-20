#!/usr/bin/env python3
"""Train a DocLayout-YOLO model from explicit local paths."""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def resolve_data(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return f"{value}.yaml"
    return str(path)

def anchor_data_path(data: str) -> str:
    """Anchor the shipped dataset configs' relative 'path:' to datasets/ in the project root (see README)."""
    config_path = Path(resolve_data(data))
    if not config_path.is_file():
        return str(config_path)
    config = yaml.safe_load(config_path.read_text())
    p = config.get('path') if isinstance(config, dict) else None
    if not p or Path(p).is_absolute():
        return str(config_path)
    candidate = (ROOT / str(p)).resolve()
    if not candidate.is_dir():
        if ROOT in config_path.resolve().parents:
            raise FileNotFoundError(
                f'Dataset directory not found: {candidate}. '
                'Download the dataset and place it under datasets/ in the project root (see README).')
        return str(config_path)
    config['path'] = str(candidate)
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        return f.name

def resolve_model(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    if path.suffix.lower() == ".pt":
        raise FileNotFoundError(
            f"A checkpoint model path must exist locally; use --pretrain for initialization: {path}"
        )
    if path.suffix.lower() in {".yaml", ".yml"}:
        return str(path)
    name = value if value.startswith("yolo") else f"yolov12{value}"
    return name if name.endswith((".yaml", ".yml", ".pt")) else f"{name}.yaml"

def parse_cache(value: str):
    value = value.lower()
    if value in {"false", "none", "off", "0"}: return False
    if value in {"true", "on", "1"}: return True
    return value

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,help='Existing dataset YAML or package dataset alias.')
    p.add_argument('--model',required=True,help='Local model YAML/.pt path or model name.')
    p.add_argument('--epoch',required=True,type=int)
    p.add_argument('--image-size',required=True,type=int)
    p.add_argument('--project',required=True,type=Path,help='External output directory.')
    p.add_argument('--name',default=None)
    p.add_argument('--pretrain',default=None,help='Existing local .pt initialization checkpoint.')
    p.add_argument('--optimizer',default='auto')
    p.add_argument('--momentum',default=0.9,type=float)
    p.add_argument('--lr0',default=0.02,type=float)
    p.add_argument('--lrf',default=0.01,type=float)
    p.add_argument('--warmup-epochs',default=3.0,type=float)
    p.add_argument('--batch-size',default=16,type=int)
    p.add_argument('--mosaic',default=1.0,type=float)
    p.add_argument('--mixup',default=0.0,type=float)
    p.add_argument('--copy-paste',default=0.0,type=float)
    p.add_argument('--cache',default='disk')
    p.add_argument('--val',default=1,type=int)
    p.add_argument('--plot',default=0,type=int)
    p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--workers',default=4,type=int)
    p.add_argument('--device',default='0')
    p.add_argument('--save-period',default=-1,type=int)
    p.add_argument('--patience',default=100,type=int)
    p.add_argument('--seed',default=0,type=int)
    p.add_argument('--box',default=7.5,type=float)
    p.add_argument('--cls',default=0.5,type=float)
    return p.parse_args()

def main():
    a=parse_args()
    data=anchor_data_path(a.data)
    model_source=resolve_model(a.model)
    if a.pretrain:
        pretrain=Path(a.pretrain).expanduser().resolve()
        if not pretrain.is_file() or pretrain.suffix.lower()!='.pt':
            raise FileNotFoundError(f'--pretrain must point to an existing local .pt file: {pretrain}')
        model_source=str(pretrain)
    model=YOLO(model_source)
    name=a.name or f'{Path(model_source).stem}_{Path(data).stem}'
    model.train(data=data,epochs=a.epoch,imgsz=a.image_size,batch=a.batch_size,
        device=a.device,workers=a.workers,project=str(a.project.expanduser()),name=name,
        optimizer=a.optimizer,momentum=a.momentum,lr0=a.lr0,lrf=a.lrf,
        warmup_epochs=a.warmup_epochs,mosaic=a.mosaic,mixup=a.mixup,
        copy_paste=a.copy_paste,cache=parse_cache(a.cache),plots=bool(a.plot),
        val=bool(a.val),resume=a.resume,save_period=a.save_period,patience=a.patience,
        seed=a.seed,box=a.box,cls=a.cls,exist_ok=False,rect=False)

if __name__=='__main__': main()
