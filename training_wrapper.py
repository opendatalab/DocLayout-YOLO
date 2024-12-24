from pathlib import Path
import argparse
from settings import LayoutParserTrainingSettings
from doclayout_yolo import YOLOv10
from datetime import datetime
import logging
from huggingface_hub import HfApi


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_model(settings: LayoutParserTrainingSettings):
    """
    Train YOLOv10 model using settings from LayoutParserTrainingSettings
    """
    # Load pretrained model
    model_path = Path(settings.local_model_dir) / settings.from_model_name
    model = YOLOv10(str(model_path))
    pretrain_name = "docstruct" if "docstruct" in settings.from_model_name else "unknown"

    # Construct run name
    name = (f"yolov10{settings.base_model}_{settings.local_data_dir}_"
           f"epoch{settings.epochs}_imgsz{settings.image_size}_"
           f"bs{settings.batch_size}_pretrain_{pretrain_name}")

    # Train model
    results = model.train(
        data=f'{settings.local_data_dir}/config.yaml',
        epochs=settings.epochs,
        warmup_epochs=settings.warmup_epochs,
        lr0=settings.lr0,
        optimizer=settings.optimizer,
        momentum=settings.momentum,
        imgsz=settings.image_size,
        mosaic=settings.mosaic,
        batch=settings.batch_size,
        device=settings.device,
        workers=settings.workers,
        plots=settings.plots,
        exist_ok=False,
        val=True,
        val_period=settings.val_period,
        resume=False,
        save_period=settings.save_period,
        patience=settings.patience,
        project=settings.local_ft_model_dir,
        name=name,
    )

    return results

def push_to_hub(
    settings: LayoutParserTrainingSettings,
    commit_message=None,
):
    """Push trained model to Hugging Face Hub"""

    # Initialize Hugging Face API
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=settings.pushed_model_repo, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repository creation failed: {e}")
        return

    # Default commit message
    if commit_message is None:
        commit_message = (
            f"Upload model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # Upload the model file
    try:
        api.upload_file(
            path_or_fileobj=settings.local_ft_model_name,
            path_in_repo=settings.pushed_model_name,
            repo_id=settings.pushed_model_repo,
            commit_message=commit_message,
        )
        print(f"Model successfully uploaded to {settings.pushed_model_repo}")
    except Exception as e:
        print(f"Upload failed: {e}")


def main(settings: LayoutParserTrainingSettings, push: bool = False, commit_message: str = None):
    
    try:
        # Train model
        logger.info(f"Starting training with batch size {settings.batch_size} and {settings.epochs} epochs")
        results = train_model(settings)
        logger.info(f"Training completed. Model saved at: {settings.local_ft_model_name}")
        
        # Push model if requested
        if args.push:
            logger.info("Pushing model to HuggingFace Hub...")
            commit_message = args.commit_message or f"Model trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            push_to_hub(
                settings=settings,
                commit_message=commit_message,
                private=True,
            )
            logger.info(f"Model successfully pushed to {settings.pushed_model_repo}")
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and optionally push YOLOv10 model')
    parser.add_argument('--push', action='store_true', help='Push model to HuggingFace Hub after training')
    parser.add_argument('--commit-message', type=str, 
                       help='Custom commit message for model push (default: timestamp)')
    args = parser.parse_args()

    settings = LayoutParserTrainingSettings()
    main(settings, args.push, args.commit_message)
