# docsynth300k pretrain
python train.py --data docsynth300k --model m-doclayout --epoch 500 --image-size 1600 --batch-size 128 --project public_dataset/docsynth300k --plot 0 --workers 1 --save-period 1 --val 0 --optimizer SGD --lr0 0.02

# D4LA from scratch
python train.py --data d4la --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/D4LA --plot 1 --optimizer SGD --lr0 0.04

# D4LA finetune
python train.py --data d4la --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/D4LA --plot 1 --optimizer SGD --lr0 0.04 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt

# D4LA evaluation
python val.py --data d4la --model checkpoint.pt --device 0 --batch-size 64

# DocLayNet from scratch
python train.py --data doclaynet --model m-doclayout --epoch 500 --image-size 1120 --batch-size 64 --project public_dataset/doclaynet --plot 1 --optimizer SGD --lr0 0.02

# DocLayNet finetune
python train.py --data doclaynet --model m-doclayout --epoch 500 --image-size 1120 --batch-size 64 --project public_dataset/doclaynet --plot 1 --optimizer SGD --lr0 0.02 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt

# DocLayNet evaluation
python val.py --data doclaynet --model checkpoint.pt --device 0 --batch-size 64
