<div align="center">

English | [简体中文](./README-zh_CN.md)


<h1>DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception</h1>

Official PyTorch implementation of [DocLayout-YOLO](https://arxiv.org/abs/2410.12628).

[![arXiv](https://img.shields.io/badge/arXiv-2405.14458-b31b1b.svg)](https://arxiv.org/abs/2410.12628) [![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97-Online%20Demo-yellow)](https://huggingface.co/spaces/opendatalab/DocLayout-YOLO) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Models%20and%20Data-yellow)](https://huggingface.co/collections/juliozhao/doclayout-yolo-670cdec674913d9a6f77b542)

</div>
    
## Abstract

> We present DocLayout-YOLO, a real-time and robust layout detection model for diverse documents, based on YOLO-v10. This model is enriched with diversified document pre-training and structural optimization tailored for layout detection. In the pre-training phase, we introduce Mesh-candidate BestFit, viewing document synthesis as a two-dimensional bin packing problem, and create a large-scale diverse synthetic document dataset, DocSynth-300K. In terms of model structural optimization, we propose a module with Global-to-Local Controllability for precise detection of document elements across varying scales. 


<p align="center">
  <img src="assets/comp.png" width=52%>
  <img src="assets/radar.png" width=44%> <br>
</p>

## News 🚀🚀🚀

**2024.10.25** 🎉🎉  **Mesh-candidate Bestfit** code is released. Mesh-candidate Bestfit is a automatic pipeline which can synthesize large-scale, high-quality, and visually appealing document layout detection dataset. Tutorial and example data are available in [here](./mesh-candidate_bestfit).

**2024.10.23** 🎉🎉  **DocSynth300K dataset** is released on [🤗Huggingface](https://huggingface.co/datasets/juliozhao/DocSynth300K), DocSynth300K is a large-scale and diverse document layout analysis pre-training dataset, which can largely boost model performance.

**2024.10.21** 🎉🎉  **Online demo** available on [🤗Huggingface](https://huggingface.co/spaces/opendatalab/DocLayout-YOLO).

**2024.10.18** 🎉🎉  DocLayout-YOLO is implemented in **[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)** for document context extraction.

**2024.10.16** 🎉🎉  **Paper** now available on [ArXiv](https://arxiv.org/abs/2410.12628).   


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


**Note:** For PDF content extraction, please refer to [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit/tree/main) and [MinerU](https://github.com/opendatalab/MinerU).

**Note:** Thanks to [NielsRogge](https://github.com/NielsRogge), DocLayout-YOLO now supports implementation directly from 🤗Huggingface, you can load model as follows:

```python
filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
model = YOLOv10(filepath)
```

or directly load using ```from_pretrained```:

```python
model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
```

more details can be found at [this PR](https://github.com/opendatalab/DocLayout-YOLO/pull/6).

**Note:** Thanks to [luciaganlulu](https://github.com/luciaganlulu), DocLayout-YOLO can perform batch inference and prediction. Instead of passing single image into ```model.predict``` in ```demo.py```, pass a **list of image path**. Besides, due to batch inference is not implemented before ```YOLOv11```, you should manually change ```batch_size``` in [here](doclayout_yolo/engine/model.py#L431).

## DocSynth300K Dataset

<p align="center">
  <img src="assets/docsynth300k.png" width=100%>
</p>

### Data Download

Use following command to download dataset(about 113G):

```python
from huggingface_hub import snapshot_download
# Download DocSynth300K
snapshot_download(repo_id="juliozhao/DocSynth300K", local_dir="./docsynth300k-hf", repo_type="dataset")
# If the download was disrupted and the file is not complete, you can resume the download
snapshot_download(repo_id="juliozhao/DocSynth300K", local_dir="./docsynth300k-hf", repo_type="dataset", resume_download=True)
```

### Data Formatting & Pre-training

If you want to perform DocSynth300K pretraining, using ```format_docsynth300k.py``` to convert original ```.parquet``` format into ```YOLO``` format. The converted data will be stored at ```./layout_data/docsynth300k```.

```bash
python format_docsynth300k.py
```

To perform DocSynth300K pre-training, use this [command](assets/script.sh#L2). We default use 8GPUs to perform pretraining. To reach optimal performance, you can adjust hyper-parameters such as ```imgsz```, ```lr``` according to your downstream fine-tuning data distribution or setting.

**Note:** Due to memory leakage in YOLO original data loading code, the pretraining on large-scale dataset may be interrupted unexpectedly, use ```--pretrain last_checkpoint.pt --resume``` to resume the pretraining process.

## Training and Evaluation on Public DLA Datasets

### Data Preparation

1. specify  the data root path

Find your ultralytics config file (for Linux user in ```$HOME/.config/Ultralytics/settings.yaml)``` and change ```datasets_dir``` to project root path.

2. Download prepared yolo-format D4LA and DocLayNet data from below and put to ```./layout_data```:

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

## Star History

If you find our project useful, please add a "star" to the repo. It's exciting to us when we see your interest, which keep us motivated to continue investing in the project!

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=opendatalab/DocLayout-YOLO&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=opendatalab/DocLayout-YOLO&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=opendatalab/DocLayout-YOLO&type=Date"
  />
</picture>

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
