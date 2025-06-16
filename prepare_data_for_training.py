import os
import uuid
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from settings import LayoutParserTrainingSettings


def prepare_data(settings: LayoutParserTrainingSettings):
    """Prepare data for YOLO training"""

    # Load dataset
    dataset = load_dataset(settings.from_dataset_repo)

    # Convert to pandas for splitting
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Create directory structure
    dirs = [
        os.path.join(settings.local_data_dir, "images", "train"),
        os.path.join(settings.local_data_dir, "images", "val"),
        os.path.join(settings.local_data_dir, "labels", "train"),
        os.path.join(settings.local_data_dir, "labels", "val"),
    ]

    # Create directories
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Process each split
    for split_name, split_data in zip(["train", "val"], [train_dataset, test_dataset]):
        for item in split_data:
            # Generate unique filename
            filename = str(uuid.uuid4())

            # Save image
            image_path = os.path.join(
                settings.local_data_dir, "images", split_name, f"{filename}.jpg"
            )
            item["image"].save(image_path)

            # Save labels
            label_path = os.path.join(
                settings.local_data_dir, "labels", split_name, f"{filename}.txt"
            )
            with open(label_path, "w") as f:
                for category, bbox in zip(
                    item["objects"]["categories"], item["objects"]["bbox"]
                ):
                    line = f"{category} {' '.join(map(str, bbox))}\n"
                    f.write(line)

    # Download YAML config
    hf_hub_download(
        repo_id=settings.from_dataset_repo,
        filename="config.yaml",
        repo_type="dataset",
        local_dir=settings.local_data_dir,
    )

    # Download pretrained model
    hf_hub_download(
        repo_id=settings.from_model_repo,
        filename=settings.from_model_name,
        repo_type="model",
        local_dir=settings.local_model_dir,
    )

    print(f"Data prepared in {settings.local_data_dir}")
    print(
        f"Train images: {len(os.listdir(os.path.join(settings.local_data_dir, 'images', 'train')))}"
    )
    print(
        f"Val images: {len(os.listdir(os.path.join(settings.local_data_dir, 'images', 'val')))}"
    )


if __name__ == "__main__":
    settings = LayoutParserTrainingSettings()
    prepare_data(settings)
