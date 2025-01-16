import os
import json
import argparse
from tqdm import tqdm


def get_map_dict(use_aug):
    """
    Get a map from a element to its corresponding save paths.

    Args:
        use_aug (bool): Whether use augmentation elements or not.
    """
    instance2pathlist = {}
    root_dir = './element_pool'
    for category in os.listdir(root_dir):
        if category == '.DS_Store':
            continue
        category_dir = os.path.join(root_dir,category)
        filelist = os.listdir(category_dir)
        for filename in tqdm(filelist):
            if filename == 'aug':
                continue
            else:
                sin_id_pathlist = []
                start_id = filename.split('.')[0]
                origin_path = os.path.join(category_dir,filename)
                sin_id_pathlist.append(origin_path)
                if 'aug' in filelist and use_aug:
                    bottom_dir = os.path.join(category_dir,f'aug/{start_id}')
                    aug_paths = os.listdir(bottom_dir)
                    aug_pathlist = [os.path.join(bottom_dir,path) for path in aug_paths]
                    sin_id_pathlist += aug_pathlist
            instance2pathlist[start_id] = sin_id_pathlist
    return instance2pathlist


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_aug', action='store_true', help="whether to use data augmentation")
    parser.add_argument('--save_path', default="./map_dict.json", type=str, help='save path for the map dict')
    args = parser.parse_args()

    map_dict = get_map_dict(use_aug=args.use_aug)

    with open(args.save_path,'w') as f:
        f.write(json.dumps(map_dict))