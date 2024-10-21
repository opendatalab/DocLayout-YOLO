# DocLayout-YOLO

Official PyTorch implementation of [DocLayout-YOLO](https://arxiv.org/abs/2410.12628).

Zhiyuan Zhao, Hengrui Kang, Bin Wang, Conghui He

[![arXiv](https://img.shields.io/badge/arXiv-2405.14458-b31b1b.svg)](https://arxiv.org/abs/2410.12628) [![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97-Online%20Demo-yellow)](https://huggingface.co/spaces/opendatalab/DocLayout-YOLO) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Models%20and%20Data-yellow)](https://huggingface.co/collections/juliozhao/doclayout-yolo-670cdec674913d9a6f77b542)

## Abstract
```text
Document Layout Analysis is crucial for real-world document understanding systems, but it encounters a challenging trade-off between speed and accuracy: multimodal methods leveraging both text and visual features achieve higher accuracy but suffer from significant latency, whereas unimodal methods relying solely on visual features offer faster processing speeds at the expense of accuracy. To address this dilemma, we introduce DocLayout-YOLO, a novel approach that enhances accuracy while maintaining speed advantages through document-specific optimizations in both pre-training and model design. For robust document pre-training, we introduce the Mesh-candidate BestFit algorithm, which frames document synthesis as a two-dimensional bin packing problem, generating the large-scale, diverse DocSynth-300K dataset. Pre-training on the resulting DocSynth-300K dataset significantly improves fine-tuning performance across various document types. In terms of model optimization, we propose a Global-to-Local Controllable Receptive Module that is capable of better handling multi-scale variations of document elements. Furthermore, to validate performance across different document types, we introduce a complex and challenging benchmark named DocStructBench. Extensive experiments on downstream datasets demonstrate that DocLayout-YOLO excels in both speed and accuracy. 
```

<p align="center">
  <img src="assets/comp.png" width=52%>
  <img src="assets/radar.png" width=44%> <br>
</p>

## Quick Start

[Online Demo](https://huggingface.co/spaces/opendatalab/DocLayout-YOLO) is now available. For local development, follow steps below:

### 1. Environment Setup

Follow these steps to set up your environment:

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

You can make predictions using either a script or the SDK:

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


For PDF content extraction, please refer to [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit/tree/main).

## Training and Evaluation on Public DLA Datasets

### Data Preparation

1. specify  the data root path

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

Training is conducted on 8 GPUs with a global batch size of 64 (8 images per device). The detailed settings and checkpoints are as follows:

| Dataset | Model | DocSynth300K Pretrained? | imgsz | Learning rate | Finetune | Evaluation | AP50 | mAP | Checkpoint |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| D4LA | DocLayout-YOLO | &cross; | 1600 | 0.04 | [command](assets/script.sh#L5) | [command](assets/script.sh#L11) | 81.7 | 69.8 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-from_scratch) |
| D4LA | DocLayout-YOLO | &check; | 1600 | 0.04 | [command](assets/script.sh#L8) | [command](assets/script.sh#L11) | 82.4 | 70.3 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-Docsynth300K_pretrained) |
| DocLayNet | DocLayout-YOLO | &cross; | 1120 | 0.02 | [command](assets/script.sh#L14) | [command](assets/script.sh#L20) | 93.0 | 77.7 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-from_scratch) |
| DocLayNet | DocLayout-YOLO | &check; | 1120 | 0.02 | [command](assets/script.sh#L17) | [command](assets/script.sh#L20) | 93.4 | 79.7 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained) |

The DocSynth300K pretrained model can be downloaded from [here](https://huggingface.co/juliozhao/DocLayout-YOLO-DocSynth300K-pretrain). Change ```checkpoint.pt``` to the path of model to be evaluated during evaluation.


## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [YOLO-v10](https://github.com/lyuwenyu/RT-DETR).

Thanks for their great work!

## Citation

```bibtex
@misc{zhao2024doclayoutyoloenhancingdocumentlayout,
      title={DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception}, 
      author={Zhiyuan Zhao and Hengrui Kang and Bin Wang and Conghui He},
      year={2024},
      eprint={2410.12628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.12628}, 
}

@article{wang2024mineru,
  title={MinerU: An Open-Source Solution for Precise Document Content Extraction},
  author={Wang, Bin and Xu, Chao and Zhao, Xiaomeng and Ouyang, Linke and Wu, Fan and Zhao, Zhiyuan and Xu, Rui and Liu, Kaiwen and Qu, Yuan and Shang, Fukai and others},
  journal={arXiv preprint arXiv:2409.18839},
  year={2024}
}

```
