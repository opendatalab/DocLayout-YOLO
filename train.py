import argparse
from doclayout_yolo import YOLOv10

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, required=True, type=str)
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--epoch', default=None, required=True, type=int)
    parser.add_argument('--optimizer', default='auto', required=False, type=str)
    parser.add_argument('--momentum', default=0.9, required=False, type=float)
    parser.add_argument('--lr0', default=0.02, required=False, type=float)
    parser.add_argument('--warmup-epochs', default=3.0, required=False, type=float)
    parser.add_argument('--batch-size', default=16, required=False, type=int)
    parser.add_argument('--image-size', default=None, required=True, type=int)
    parser.add_argument('--mosaic', default=1.0, required=False, type=float)
    parser.add_argument('--pretrain', default=None, required=False, type=str)
    parser.add_argument('--val', default=1, required=False, type=int)
    parser.add_argument('--val-period', default=1, required=False, type=int)
    parser.add_argument('--plot', default=0, required=False, type=int)
    parser.add_argument('--project', default=None, required=True, type=str)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction)
    parser.add_argument('--workers', default=4, required=False, type=int)
    parser.add_argument('--device', default="0,1,2,3,4,5,6,7", required=False, type=str)
    parser.add_argument('--save-period', default=10, required=False, type=int)
    parser.add_argument('--patience', default=100, required=False, type=int)
    args = parser.parse_args()
    
    # using '.pt' will load pretrained model
    if args.pretrain is not None:
        if args.pretrain == 'coco':
            model = f'yolov10{args.model}.pt'
            pretrain_name = 'coco'
        elif 'pt' in args.pretrain:
            model = args.pretrain
            if 'bestfit' in args.pretrain:
                pretrain_name = 'bestfit_layout'
            else:
                pretrain_name = "unknown"
        else:
            raise BaseException("Wrong pretrained model specified!")
    else:
        model = f'yolov10{args.model}.yaml'
        pretrain_name = 'None'
    
    # Load a pre-trained model
    model = YOLOv10(model)
    
    # whether to val during training
    if args.val:
        val = True
    else:
        val = False
        
    # whether to plot
    if args.plot:
        plot = True
    else:
        plot = False
    
    # Train the model
    name = f"yolov10{args.model}_{args.data}_epoch{args.epoch}_imgsz{args.image_size}_bs{args.batch_size}_pretrain_{pretrain_name}"
    results = model.train(
        data=f'{args.data}.yaml', 
        epochs=args.epoch, 
        warmup_epochs=args.warmup_epochs,
        lr0=args.lr0,
        optimizer=args.optimizer,
        momentum=args.momentum,
        imgsz=args.image_size, 
        mosaic=args.mosaic,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        plots=plot,
        exist_ok=False,
        val=val,
        val_period=args.val_period,
        resume=args.resume,
        save_period=args.save_period,
        patience=args.patience,
        project=args.project, 
        name=name,
    )