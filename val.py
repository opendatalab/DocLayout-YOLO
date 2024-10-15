import argparse
from doclayout_yolo import YOLOv10

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, required=True, type=str)
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--batch-size', default=16, required=False, type=int)
    parser.add_argument('--device', default="0,1,2,3,4,5,6,7", required=False, type=str)
    args = parser.parse_args()
    
    # Load a pre-trained model
    model = YOLOv10(args.model)
    
    # Train the model
    model.val(
        data=f'{args.data}.yaml', 
        batch=args.batch_size,
        device=args.device,
    )