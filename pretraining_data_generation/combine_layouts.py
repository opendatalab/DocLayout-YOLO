import os
import json
import argparse
from tqdm import tqdm


def combine_layouts(seperate_layouts_dir):
    """
    Combining seperate layouts into one json.

    Args:
        seperate_layouts_dir (str): Directory to save seperate layouts json files generated by bestfit_generator.py
    """
    combined_layouts = []
    for item in tqdm(os.listdir(seperate_layouts_dir),desc='Combining seperate layouts'):
        abs_path = os.path.join(seperate_layouts_dir,item)
        json_file = json.load(open(abs_path))
        combined_layouts.append(json_file)
    return combined_layouts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seperate_layouts_dir', default="./generate_layouts/seperate", type=str, help="directory to save seperate layouts")
    parser.add_argument('--save_path', default="./generate_layouts/combined_layouts.json", type=str, help='save path for combined generated layouts')
    args = parser.parse_args()

    combined_layouts = combine_layouts(seperate_layouts_dir=args.seperate_layouts_dir)

    with open(args.save_path,'w') as f:
        f.write(json.dumps(combined_layouts,indent=4))