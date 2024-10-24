## Pretraining Data Generation

<p align="center">
  <img src="assets/mesh-candidate_bestfit.png" width=100%> <br>
</p>

You can generate a large scale of diverse data for pretraining applying our proposed method, just follow steps below:

### 1. Environment Setup

You need to install [PyMuPDF](https://pypi.org/project/PyMuPDF/1.23.7/) for subsequent rendering via pip:

```bash
cd pretraining_data_generation
pip install pymupdf==1.23.7
```

### 2. Dataset Preparation

Taking M6Doc dataset as an example, the required file structure of element pool is as follows. The first-level subdirectories are named after the specific categories, and the elements inside are named with corresponding annotation ids in the raw json file of the dataset. Organized M6Doc element pool can be downloaded from [Google drive](https://drive.google.com/file/d/1MrIFObKr1bDGgZLBQM_c_Dvobkp6mjFE/view?usp=sharing).

```bash
./element_pool
├── advertisement
│   ├── 727.jpg
│   ├── 919.jpg
│   ├── 1423.jpg
│   └── ...
├── algorithm
│   ├── 12653.jpg
│   ├── 17485.jpg
│   ├── 44364.jpg
│   └── ...
└── ...
```

### 3. Data Augmentation(Optional)

If you want to apply our designed augmentation pipeline to your element pool, you can just run:

```bash
python augmentation.py --min_count 100 --aug_times 50
```

### 4. Map Dict

To facilitate the random selection of candidates during the rendering phase, it is necessary to establish a mapping from elements to all of their candidate paths:

```bash
python map_dict.py --save_path ./map_dict.json --use_aug
```

### 5. Layout Generation

Now, you can generate diverse layouts using Mesh-candidate Bestfit algorithm. To prevent process blocking, it will save the result of each layout in a timely manner, but you can use the script below to combine them all:

```bash
python bestfit_generator.py --generate_num 10000 --json_path ./M6Doc.json --output_dir ./generated_layouts/seperate
python combine_layouts.py --seperate_layouts_dir ./generate_layouts/seperate --save_path ./generate_layouts/combined_layouts.json
```

### 6. Rendering

Finally, you can render generated layouts and save the results in yolo format via the script below:

```bash
python rendering.py --json_path ./generate_layouts/combined_layouts.json --map_dict_path ./map_dict.json --save_dir ./generated_dataset 
```