## Mesh-candidate Bestfit

<p align="center">
  <img src="../assets/Mesh-candidate Bestfit.png" width=100%> <br>
</p>

You can generate a large scale of diverse data for pretraining applying our proposed method Mesh-candidate Bestfit, just follow steps below:

### 1. Environment Setup

You need to install [PyMuPDF](https://pypi.org/project/PyMuPDF/1.23.7/) for subsequent rendering via pip:

```bash
cd "Mesh-candidate Bestfit"
pip install pymupdf==1.23.7
```

### 2. Preprocessing

- **Data Preparation**

  Two primary things need to be well prepared before starting generation: 

  1. Original annotation file of your dataset

    - It must be a json following COCO format.
    - Each instance has a unique instance id.
    - It should be placed under `./` folder

  2. Element Pool

  You can easily extract elements of different categories based on the original annotation file. However, it is required to be structured like this:

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

  The first-level subdirectories are named after the specific categories, and the elements inside are named with corresponding instance ids in the raw json file of the dataset. 

  **Note:** For convenience, we provide original annotation file and element pool for M6Doc dataset, which can be downloaded from [annotation file](https://drive.google.com/file/d/1ua41Gs3UW8iuoJp21tZ4-lczVrcEm-gP/view?usp=sharing) and [element pool](https://drive.google.com/file/d/1MrIFObKr1bDGgZLBQM_c_Dvobkp6mjFE/view?usp=sharing), respectively.

- **Data Augmentation(Optional)**

  If you want to apply our designed augmentation pipeline to your element pool, you can just run:

  ```bash
  python augmentation.py --min_count 100 --aug_times 50
  ```

  The script will perform augmentation pipeline `aug_times` times on each element of categories whose element number is less than `min_count`.

- **Map Dict**

  To facilitate the random selection of candidates during the rendering phase, it is necessary to establish a mapping from elements to all of their candidate paths:

  ```bash
  python map_dict.py --save_path ./map_dict.json --use_aug
  ```

### 3. Layout Generation

Now, you can generate diverse layouts using Mesh-candidate Bestfit algorithm. To prevent process blocking, it will save the result of each layout in a timely manner, but you can use the [combine_layouts.py](./combine_layouts.py) script to combine them all together like this:

```bash
python bestfit_generator.py --generate_num 10000 --json_path ./M6Doc.json --output_dir ./generated_layouts/seperate
python combine_layouts.py --seperate_layouts_dir ./generate_layouts/seperate --save_path ./generate_layouts/combined_layouts.json
```

Afterwards, feel free to delete the seperate layouts since they are no longer used.

### 4. Rendering

Finally, you can render generated layouts and save the results in yolo format via the script below:

```bash
python rendering.py --json_path ./generate_layouts/combined_layouts.json --map_dict_path ./map_dict.json --save_dir ./generated_dataset 
```

### Visualization

We provide [visualize.ipynb](./visualize.ipynb) to visualize the layouts generated by our proposed methods. Here, we display some generation cases below:

<p align="center">
  <img src="../assets/visualization.png" width=100%> <br>
</p>