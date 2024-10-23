import io
import os
import tqdm
import pandas as pd
from PIL import Image

parquet_list = os.listdir("./docsynth300k")
parquet_list = [f for f in parquet_list if "parquet" in f]

save_path = "./layout_data/docsynth300k"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(os.path.join(save_path, "images"))
    os.makedirs(os.path.join(save_path, "labels"))
train_txt = open(os.path.join(save_path, "train300k.txt"), "w")
    
for parquet in tqdm.tqdm(parquet_list):
    df = pd.read_parquet(os.path.join("docsynth300k", parquet))
    for index, row in tqdm.tqdm(df.iterrows()):
        filename = row['filename']
        image_data = row['image_data']
        anno_string = row['anno_string']
        image = Image.open(io.BytesIO(image_data))
        # save image / anno / txt
        train_txt.write(os.path.join("images", filename) + "\n")
        image.save(os.path.join(save_path, "images", filename))
        with open(os.path.join(save_path, "labels", filename.replace(".jpg", ".txt")), "w") as f:
            for line in anno_string:
                f.write(line)
                
# write validation file (dummy)
train_txt.close()
image_list = list(open(os.path.join(save_path, "train300k.txt"), "r").readlines())[:1000]
val = open(os.path.join(save_path, "val.txt"), "w")
for line in image_list:
    val.write(line)
val.close()