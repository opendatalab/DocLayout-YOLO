# DocLayout-YOLO: Advancing Document Layout Analysis with Mesh-candidate Bestfit and Global-to-local perception


Official PyTorch implementation of **DocLayout-YOLO**.

Zhiyuan Zhao, Hengrui Kang, Bin Wang, Conghui He

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
We introduce DocLayout-YOLO, which not only enhances accuracy but also preserves the speed advantage through optimization from pre-training and model perspectives in a document-tailored manner. In terms of robust document pretraining, we innovatively regard document synthetic as a 2D bin packing problem and introduce Mesh-candidate Bestfit, which enables the generation of large-scale, diverse document datasets. The model, pre-trained on the resulting DocSynth300K dataset, significantly enhances fine-tuning performance across a variety of document types. In terms of model enhancement for document understanding, we propose a Global-to-local Controllable Receptive Module which emulates the human visual process from global to local perspectives and features a controllable module for feature extraction and integration. Experimental results on extensive downstream datasets show that the proposed DocLayout-YOLO excels at both speed and accuracy.
</details>

<p align="center">
  <img src="assets/comp.png" width=52%>
  <img src="assets/radar.png" width=44%> <br>
</p>

## Quick Start

### 1. Environment Setup

To set up your environment, follow these steps:

```bash
conda create -n doclayout_yolo python=3.10
conda activate doclayout_yolo
pip install -e .
```

**Note:** If you only need the package for inference, you can simply install it via pip:

```bash
pip install doclayout-yolo
```

### 2. Prediction

You can perform predictions using either a script or the SDK:

- **Script**

  Run the following command to make a prediction using the script:

  ```bash
  python demo.py --model path/to/model --image-path path/to/image
  ```

- **SDK**

  Here is an example of how to use the SDK for prediction:

  ```python
  import cv2
  from doclayout_yolo import YOLOv10

  # Load the pre-trained model
  model = YOLOv10("path/to/provided/model")

  # Perform prediction
  det_res = model.predict(
      "path/to/image",   # Image to predict
      imgsz=1024,        # Prediction image size
      conf=0.2,          # Confidence threshold
      device="cuda:0"    # Device to use (e.g., 'cuda:0' or 'cpu')
  )

  # Annotate and save the result
  annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
  cv2.imwrite("result.jpg", annotated_frame)
  ```


We provide model fine-tuned on **DocStructBench** for prediction, **which is capable of handing various document types**. Model can be downloaded from [here](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/tree/main) and example images can be found under ```assets/example```.

<p align="center">
  <img src="assets/showcase.png" width=100%> <br>
</p>


You also can use ```predict_single.py``` for prediction with custom inference settings. For batch process, please refer to [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit/tree/main).

## Training and Evaluation on Public DLA Datasets

### Data Preparation

1. specify data root path

Find your ultralytics config file (for Linux user in ```$HOME/.config/Ultralytics/settings.yaml)``` and change ```datasets_dir``` to project root path.

2. Download prepared yolo-format D4LA and doclaynet data from below and put to ```./layout_data```:

| Dataset | Download |
|:--:|:--:|
| D4LA | [link](https://huggingface.co/datasets/juliozhao/doclayout-yolo-D4LA) |
| DocLayNet | [link](https://huggingface.co/datasets/juliozhao/doclayout-yolo-DocLayNet) |

the file structure is as follows:

```bash
./layout_data
├── D4LA
│   ├── images
│   ├── labels
│   ├── test.txt
│   └── train.txt
└── doclaynet
    ├── images
    ├── labels
    ├── val.txt
    └── train.txt
```

### Training and Evaluation

Training is conducted on 8 GPUs with a global batch size of 64 (8 images per device), detailed settings and checkpoints are as follows:

| Dataset | Model | DocSynth300K Pretrained? | imgsz | Learning rate | Finetune | Evaluation | AP50 | mAP | Checkpoint |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| D4LA | DocLayout-YOLO | &cross; | 1600 | 0.04 | [command](assets/script.sh#L5) | [command](assets/script.sh#L11) | 81.7 | 69.8 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-from_scratch) |
| D4LA | DocLayout-YOLO | &check; | 1600 | 0.04 | [command](assets/script.sh#L8) | [command](assets/script.sh#L11) | 82.4 | 70.3 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-Docsynth300K_pretrained) |
| DocLayNet | DocLayout-YOLO | &cross; | 1120 | 0.02 | [command](assets/script.sh#L14) | [command](assets/script.sh#L20) | 93.0 | 77.7 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-from_scratch) |
| DocLayNet | DocLayout-YOLO | &check; | 1120 | 0.02 | [command](assets/script.sh#L17) | [command](assets/script.sh#L20) | 93.4 | 79.7 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained) |

The DocSynth300K pretrained model can be downloaded from [here](https://huggingface.co/juliozhao/DocLayout-YOLO-DocSynth300K-pretrain). Change ```checkpoint.pt``` to the path of model to be evaluated during evaluation.

## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [YOLO-v10](https://github.com/lyuwenyu/RT-DETR).

Thanks for these great work!
