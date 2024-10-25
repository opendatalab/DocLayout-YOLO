import os
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
import albumentations as A


class EdgeDetection(A.ImageOnlyTransform):
    """
    A class for edge extraction of images with the sobel filter
    """
    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        mag = mag / np.max(mag) * 255
        return np.uint8(mag)


def pipeline(h, w):
    """
    Whole data augmentation pipeline with the input of image size

    Args:
        h (float): Height of the image.
        w (float): Width of the image.
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.RandomResizedCrop(height=h, width=w, scale=(0.5, 0.9), ratio=(w / h, w / h), p=0.7),  # keep h/w ratio the same
        EdgeDetection(p=0.2),
        A.ElasticTransform(alpha_affine=5, p=0.2),
        A.GaussNoise(var_limit=(100, 1200), p=1),
    ])


def perform_augmentation(img, transform, save_dir, prefix_id, aug_times):
    """
    Perform augmentation for a single element with many times.

    Args:
        img (image): An elment.
        transform (sequence): Data augmentation pipeline.
        save_dir (str): Root directory to save.
        prefix_id (str): Raw id for the element.
        aug_times (int): Augmentation times.
    """
    for _ in range(aug_times):
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

        suffix_id = str(time.time()).replace(".", "_")
        prefix_id_dir = os.path.join(save_dir, prefix_id)
        os.makedirs(prefix_id_dir, exist_ok=True)
        aug_element_path = os.path.join(prefix_id_dir, f'{prefix_id}_{suffix_id}.jpg')
        cv2.imwrite(aug_element_path, transformed_image_bgr)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform Image Augmentation")
    parser.add_argument("--min_count", type=int, default=100, help="Minimum number of elements for categories that do not require data augmentation")
    parser.add_argument("--aug_times", type=int, default=50, help="Number of augmentations per element")
    args = parser.parse_args()

    root_dir = './element_pool'

    for category in tqdm(os.listdir(root_dir),desc='Categories done'):
        category_dir = os.path.join(root_dir,category)
        if len(os.listdir(category_dir)) > args.min_count:
            continue
        else:
            save_dir = os.path.join(category_dir, 'aug')
            for raw_element in os.listdir(category_dir):
                raw_element_path = os.path.join(category_dir, raw_element)
                img = cv2.imread(raw_element_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                element_id = raw_element.split('.')[0]

                transform = pipeline(h, w)

                perform_augmentation(img=img,transform=transform,save_dir=save_dir,prefix_id=element_id,aug_times=args.aug_times)